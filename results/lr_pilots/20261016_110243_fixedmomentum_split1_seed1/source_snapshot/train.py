import argparse
import csv
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from config import ExperimentConfig
from models.resnet import build_model
from schedulers import build_scheduler
from utils.logger import CsvLogger
from utils.metrics import accuracy
from utils.seed import set_seed


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="FT5 CIFAR-10 learning-rate experiment")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--plot-dir", default="plots")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--scheduler", default="constant")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--validation-only", action="store_true", help="Skip test evaluation during tuning.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quiet", action="store_true", help="Disable per-batch progress bars.")
    parser.add_argument("--config", type=Path, help="Frozen JSON protocol; explicit CLI values override defaults.")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        with preliminary.config.open(encoding="utf-8") as handle:
            frozen = json.load(handle)
        names = ("epochs", "batch_size", "split_seed", "validation_size", "num_workers", "device", "data_dir", "model")
        parser.set_defaults(**{name: frozen[name] for name in names if name in frozen})
        settings = frozen.get("schedulers", {}).get(preliminary.scheduler)
        if settings is None:
            parser.error("Scheduler is absent from the frozen configuration.")
        parser.set_defaults(lr=settings["lr"])
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or not 0 < args.validation_size < 50000:
        parser.error("Invalid epochs, batch size, num_workers or validation size.")

    return ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        plot_dir=args.plot_dir,
        run_name=args.run_name,
        model=args.model,
        scheduler=args.scheduler,
        epochs=args.epochs,
        batch_size=args.batch_size,
        initial_lr=args.lr,
        seed=args.seed,
        split_seed=args.split_seed,
        validation_only=args.validation_only,
        num_workers=args.num_workers,
        validation_size=args.validation_size,
        device=args.device,
        quiet=args.quiet,
    )


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_full = datasets.CIFAR10(root=config.data_dir, train=True, download=False, transform=train_transform)
    val_full = datasets.CIFAR10(root=config.data_dir, train=True, download=False, transform=eval_transform)
    test_set = datasets.CIFAR10(root=config.data_dir, train=False, download=False, transform=eval_transform)

    generator = torch.Generator().manual_seed(config.split_seed)
    permutation = torch.randperm(len(train_full), generator=generator).tolist()
    val_indices = permutation[: config.validation_size]
    train_indices = permutation[config.validation_size :]

    train_set = Subset(train_full, train_indices)
    val_set = Subset(val_full, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scheduler_step="epoch", quiet=False):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for inputs, targets in tqdm(loader, leave=False, ascii=True, disable=quiet):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_train:
                loss.backward()
                optimizer.step()
                if scheduler is not None and scheduler_step == "batch":
                    scheduler.step()

            correct, seen = accuracy(outputs, targets)
            total_loss += loss.item() * seen
            total_correct += correct
            total_seen += seen

    return total_loss / total_seen, total_correct / total_seen


def append_summary(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, [])
        if existing_header != list(row):
            archived_path = path.with_name(f"{path.stem}_legacy_{int(time.time())}{path.suffix}")
            path.rename(archived_path)
            exists = False
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = select_device(config.device)

    run_date = datetime.now().astimezone().isoformat(timespec="seconds")
    git_commit = get_git_commit()
    experiment_id = f"{config.run_name}_{config.scheduler}_seed{config.seed}"
    run_id = experiment_id
    output_dir = Path(config.output_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders(config)
    model = build_model(config.model, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.initial_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler, scheduler_step, scheduler_metadata = build_scheduler(
        config.scheduler, optimizer, epochs=config.epochs, steps_per_epoch=len(train_loader)
    )
    train_size = len(train_loader.dataset)
    validation_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    split_record = {"split_seed": config.split_seed, "train_indices": train_loader.dataset.indices,
                    "validation_indices": val_loader.dataset.indices}
    split_hash = hashlib.sha256(json.dumps(split_record, sort_keys=True).encode()).hexdigest()
    write_json(output_dir / f"{run_id}_split.json", split_record)
    source_hash = hashlib.sha256()
    for source in ("train.py", "config.py", "schedulers.py", "models/resnet.py", "utils/seed.py",
                   "utils/logger.py", "utils/metrics.py"):
        source_hash.update(source.encode())
        source_hash.update((Path(__file__).parent / source).read_bytes())

    epoch_logger = CsvLogger(
        output_dir / f"{run_id}_epochs.csv",
        [
            "epoch",
            "lr",
            "momentum",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "epoch_seconds",
        ],
    )

    best_val_acc = -1.0
    final_val_acc = 0.0
    best_checkpoint = checkpoint_dir / f"{run_id}_best.pt"
    start_time = time.perf_counter()
    metadata = {
        "experiment_id": experiment_id,
        "date": run_date,
        "git_commit": git_commit,
        "dataset": config.dataset,
        "architecture": config.model,
        "optimizer": config.optimizer,
        "scheduler": scheduler_metadata["scheduler_name"],
        "scheduler_params": scheduler_metadata["scheduler_params"],
        "scheduler_step": scheduler_metadata["scheduler_step"],
        "initial_lr": scheduler_metadata["initial_lr"],
        "minimum_lr": scheduler_metadata["min_lr"],
        "maximum_lr": scheduler_metadata["max_lr"],
        "batch_size": config.batch_size,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "epochs": config.epochs,
        "seed": config.seed,
        "split_seed": config.split_seed,
        "split_sha256": split_hash,
        "source_sha256": source_hash.hexdigest(),
        "protocol_version": "fixed-momentum-fixed-split-v2",
        "train_size": train_size,
        "validation_size": validation_size,
        "test_size": test_size,
        "device": str(device),
        "config": asdict(config),
    }
    metadata_path = output_dir / f"{run_id}_metadata.json"
    write_json(metadata_path, metadata)

    print(f"Device: {device}")
    print(f"Run: {run_id}")
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler, scheduler_step, config.quiet
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, quiet=config.quiet)
        final_val_acc = val_acc
        if scheduler is not None and scheduler_step == "epoch":
            scheduler.step()

        epoch_seconds = time.perf_counter() - epoch_start
        epoch_logger.write(
            {
                "epoch": epoch,
                "lr": current_lr,
                "momentum": optimizer.param_groups[0]["momentum"],
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_seconds": epoch_seconds,
            }
        )
        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"lr={current_lr:.6f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "metadata": metadata,
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                },
                best_checkpoint,
            )

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = (None, None) if config.validation_only else run_epoch(
        model, test_loader, criterion, device, quiet=config.quiet
    )
    total_seconds = time.perf_counter() - start_time

    append_summary(
        output_dir / "results.csv",
        {
            "experiment_id": experiment_id,
            "date": run_date,
            "git_commit": git_commit,
            "dataset": config.dataset,
            "architecture": config.model,
            "optimizer": config.optimizer,
            "scheduler": scheduler_metadata["scheduler_name"],
            "scheduler_params": scheduler_metadata["scheduler_params"],
            "initial_lr": scheduler_metadata["initial_lr"],
            "minimum_lr": scheduler_metadata["min_lr"],
            "maximum_lr": scheduler_metadata["max_lr"],
            "batch_size": config.batch_size,
            "momentum": config.momentum,
            "weight_decay": config.weight_decay,
            "epochs": config.epochs,
            "seed": config.seed,
            "split_seed": config.split_seed,
            "split_sha256": split_hash,
            "source_sha256": source_hash.hexdigest(),
            "protocol_version": "fixed-momentum-fixed-split-v2",
            "best_val_acc": best_val_acc,
            "final_val_acc": final_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "training_time_seconds": total_seconds,
            "checkpoint_path": str(best_checkpoint),
            "metadata_path": str(metadata_path),
        },
    )
    print(f"Finished {run_id}: best_val_acc={best_val_acc:.4f}, test_acc={test_acc}")


if __name__ == "__main__":
    main()

