import argparse
import csv
import time
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
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--scheduler", default="constant")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quiet", action="store_true", help="Disable per-batch progress bars.")
    args = parser.parse_args()

    return ExperimentConfig(
        data_dir=args.data_dir,
        run_name=args.run_name,
        model=args.model,
        scheduler=args.scheduler,
        epochs=args.epochs,
        batch_size=args.batch_size,
        initial_lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
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

    generator = torch.Generator().manual_seed(config.seed)
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
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = select_device(config.device)

    run_id = f"{config.run_name}_{config.scheduler}_seed{config.seed}"
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
    scheduler, scheduler_step = build_scheduler(
        config.scheduler, optimizer, epochs=config.epochs, steps_per_epoch=len(train_loader)
    )

    epoch_logger = CsvLogger(
        output_dir / f"{run_id}_epochs.csv",
        [
            "epoch",
            "lr",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "epoch_seconds",
        ],
    )

    best_val_acc = 0.0
    best_checkpoint = checkpoint_dir / f"{run_id}_best.pt"
    start_time = time.perf_counter()

    print(f"Device: {device}")
    print(f"Run: {run_id}")
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler, scheduler_step, config.quiet
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, quiet=config.quiet)
        if scheduler is not None and scheduler_step == "epoch":
            scheduler.step()

        epoch_seconds = time.perf_counter() - epoch_start
        epoch_logger.write(
            {
                "epoch": epoch,
                "lr": current_lr,
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
                    "config": vars(config),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                },
                best_checkpoint,
            )

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device, quiet=config.quiet)
    total_seconds = time.perf_counter() - start_time

    append_summary(
        output_dir / "results.csv",
        {
            "run_id": run_id,
            "dataset": config.dataset,
            "model": config.model,
            "optimizer": config.optimizer,
            "scheduler": config.scheduler,
            "initial_lr": config.initial_lr,
            "batch_size": config.batch_size,
            "weight_decay": config.weight_decay,
            "epochs": config.epochs,
            "seed": config.seed,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "training_time_seconds": total_seconds,
            "checkpoint_path": str(best_checkpoint),
        },
    )
    print(f"Finished {run_id}: best_val_acc={best_val_acc:.4f}, test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()

