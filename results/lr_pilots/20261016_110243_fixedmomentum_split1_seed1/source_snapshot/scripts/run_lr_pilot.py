"""Validation-only LR range test and equal-budget scheduler pilots."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ExperimentConfig
from models.resnet import build_model
from schedulers import build_scheduler
from train import build_loaders, select_device, write_json
from utils.seed import set_seed

SCHEDULERS = ["constant", "step", "multistep", "cosine", "warmrestart", "cyclic", "onecycle", "warmup_cosine"]


def save_schedules(root, selected):
    directory = root / "formal_schedules"
    directory.mkdir()
    metadata = {}
    fig, ax = plt.subplots(figsize=(11, 6))
    for row in selected.itertuples():
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=row.candidate_lr,
                                    momentum=0.9, weight_decay=5e-4)
        scheduler, interval, info = build_scheduler(row.scheduler, optimizer, epochs=150, steps_per_epoch=352)
        metadata[row.scheduler] = {"lr": float(row.candidate_lr), **info}
        records = []
        count = 150 * 352 if interval == "batch" else 150
        for step in range(count):
            epoch = step / 352 + 1 if interval == "batch" else step + 1
            records.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                            "momentum": optimizer.param_groups[0]["momentum"]})
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        data = pd.DataFrame(records)
        data.to_csv(directory / f"{row.scheduler}.csv", index=False)
        ax.plot(data.epoch, data.lr, label=row.scheduler)
    ax.set(xlabel="Epoch", ylabel="LR used before optimizer update", title="Frozen 150-epoch LR schedules (momentum=0.9)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "lr_comparison.png", dpi=160)
    plt.close(fig)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-epochs", type=int, default=3)
    parser.add_argument("--range-steps", type=int, default=704)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.pilot_epochs < 3 or args.range_steps < 100:
        parser.error("Use at least 3 pilot epochs and 100 range steps.")
    project = Path(__file__).resolve().parents[1]
    root = project / "results" / "lr_pilots" / datetime.now().strftime("%Y%m%d_%H%M%S_fixedmomentum_split1_seed1")
    root.mkdir(parents=True)
    print(f"OUTPUT={root}", flush=True)
    source_dir = root / "source_snapshot"
    source_dir.mkdir()
    hashes = {}
    for name in ["train.py", "config.py", "schedulers.py", "models/resnet.py", "utils/seed.py", "utils/logger.py", "utils/metrics.py", "scripts/run_lr_pilot.py"]:
        content = (project / name).read_bytes()
        target = source_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    write_json(root / "environment.json", {"python": sys.version, "torch": torch.__version__,
               "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"})
    config = ExperimentConfig(data_dir=args.data_dir, seed=1, split_seed=1, device=args.device)
    device = select_device(args.device)
    set_seed(1)
    train, validation, _ = build_loaders(config)
    model = build_model("resnet18", num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    records = []
    smooth = 0.0
    best = float("inf")
    iterator = iter(train)
    model.train()
    for step in range(args.range_steps):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(train)
            inputs, targets = next(iterator)
        lr = 1e-4 * (1e4 ** (step / (args.range_steps - 1)))
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs.to(device)), targets.to(device))
        value = loss.item()
        if not math.isfinite(value):
            break
        smooth = 0.98 * smooth + 0.02 * value
        corrected = smooth / (1 - 0.98 ** (step + 1))
        records.append({"step": step + 1, "lr": lr, "loss": value, "smoothed_loss": corrected})
        if step > 50 and corrected > 4 * best:
            break
        best = min(best, corrected)
        loss.backward()
        optimizer.step()
    data = pd.DataFrame(records)
    if len(data) < 100:
        raise RuntimeError("LR range test failed before sufficient observations.")
    data.to_csv(root / "lr_range.csv", index=False)
    minimum = data.iloc[50:].sort_values("smoothed_loss").iloc[0]
    # Two conservative grid values below the observed smoothed-loss minimum.
    grid = [0.01, 0.03, 0.1, 0.3]
    high_index = max(1, max((i for i, lr in enumerate(grid) if lr <= float(minimum.lr)), default=1))
    candidates = grid[high_index - 1:high_index + 1]
    write_json(root / "pilot_protocol.json", {
        "seed": 1, "split_seed": 1, "momentum": 0.9, "weight_decay": 5e-4,
        "batch_size": 128, "pilot_epochs": args.pilot_epochs, "range_steps_requested": args.range_steps,
        "range_steps_completed": len(data), "range_minimum_lr": float(minimum.lr),
        "candidates": candidates, "candidate_rule": "Adjacent grid values at/below smoothed-loss minimum; minimum grid bracket 0.01/0.03",
        "selection": "Highest final validation accuracy; tie: lower final validation loss, then lower LR",
        "test_evaluation": False, "schedulers": SCHEDULERS,
        "scope": "Short-horizon pilot only; not proof of optimal 150-epoch hyperparameters",
    })
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(data.lr, data.smoothed_loss)
    for lr in candidates:
        ax.axvline(lr, linestyle="--", label=f"candidate {lr}")
    ax.set(xlabel="Learning rate", ylabel="Smoothed training loss", title="LR range test (training only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "lr_range.png", dpi=160)
    plt.close(fig)
    del model, optimizer, train, validation
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summaries = []
    for scheduler in SCHEDULERS:
        for lr in candidates:
            run_dir = root / scheduler / f"lr{str(lr).replace('.', 'p')}"
            command = [sys.executable, "-u", "train.py", "--scheduler", scheduler, "--lr", str(lr),
                       "--epochs", str(args.pilot_epochs), "--seed", "1", "--split-seed", "1",
                       "--batch-size", "128", "--device", args.device, "--data-dir", args.data_dir,
                       "--run-name", "pilot_v2", "--validation-only", "--quiet",
                       "--output-dir", str(run_dir / "results"), "--checkpoint-dir", str(run_dir / "checkpoints")]
            print(f"PILOT {scheduler} lr={lr}", flush=True)
            run_dir.mkdir(parents=True)
            with (run_dir / "console.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = pd.read_csv(run_dir / "results" / "results.csv").iloc[0].to_dict()
            epochs = pd.read_csv(next((run_dir / "results").glob("*_epochs.csv")))
            summary["candidate_lr"] = lr
            summary["final_val_loss"] = float(epochs.iloc[-1].val_loss)
            summaries.append(summary)
            pd.DataFrame(summaries).to_csv(root / "pilot_summary.csv", index=False)
    summary = pd.DataFrame(summaries)
    for name, digest in hashes.items():
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during pilots: {name}; cannot freeze protocol.")
    if summary.source_sha256.nunique() != 1 or summary.split_sha256.nunique() != 1:
        raise RuntimeError("Pilot sources or dataset splits are inconsistent.")
    selected = summary.sort_values(["final_val_acc", "final_val_loss", "candidate_lr"], ascending=[False, True, True]).groupby("scheduler", sort=False).head(1)
    selected.to_csv(root / "selected_parameters.csv", index=False)
    frozen = {
        "protocol_version": "fixed-momentum-fixed-split-v2", "status": "frozen_for_main_comparison",
        "pilot_dir": str(root), "selection_metric": "final_val_acc", "epochs": 150,
        "seeds": [1, 2, 3], "split_seed": 1, "validation_size": 5000, "batch_size": 128,
        "dataset": "cifar10", "model": "resnet18", "optimizer": "sgd", "momentum": 0.9,
        "weight_decay": 5e-4, "cycle_momentum": False, "num_workers": 0,
        "device": args.device, "data_dir": args.data_dir,
        "split_sha256": summary.iloc[0].split_sha256,
        "source_sha256": summary.iloc[0].source_sha256,
        "schedulers": save_schedules(root, selected),
        "loss": "CrossEntropyLoss", "training_augmentation": ["RandomCrop(32,padding=4)", "RandomHorizontalFlip"],
        "evaluation_augmentation": "none", "normalization_mean": [0.4914, 0.4822, 0.4465],
        "normalization_std": [0.2023, 0.1994, 0.2010],
        "schedule_rule": "schedulers.py; epoch-based parameters scale with the 150-epoch budget",
        "limitations": f"One tuning seed, two candidates and {args.pilot_epochs} pilot epochs; rankings may change at 150 epochs. No test data used.",
    }
    frozen["source_hashes"] = hashes
    write_json(root / "formal_config.json", frozen)
    fig, ax = plt.subplots(figsize=(11, 5))
    for lr, group in summary.groupby("candidate_lr"):
        ax.plot(group.scheduler, group.final_val_acc, marker="o", label=f"LR parameter={lr}")
    ax.set(xlabel="Scheduler", ylabel="Final validation accuracy", title=f"Equal-budget pilots: {args.pilot_epochs} epochs, seed 1")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "pilot_validation_comparison.png", dpi=160)
    plt.close(fig)
    table = selected[["scheduler", "candidate_lr", "final_val_acc", "final_val_loss"]].to_csv(index=False)
    (root / "README.md").write_text(
        "# LR pilot and frozen protocol v2\n\n"
        "Training-only exponential LR range test; equal-budget validation-only pilots.\n"
        f"Each method: {len(candidates)} candidates x {args.pilot_epochs} epochs, seed=1, split_seed=1.\n"
        "Momentum=0.9, weight decay=5e-4; momentum cycling disabled.\n\n"
        "Selection uses final validation accuracy, then validation loss, then lower LR. Test set was not evaluated.\n"
        "These short pilots are an economical screening, not a definitive hyperparameter search.\n"
        "Schedule durations scale to 150 epochs in the frozen configuration.\n\n"
        "![LR range](lr_range.png)\n\n![Validation comparison](pilot_validation_comparison.png)\n\n"
        "![Frozen formal schedules](formal_schedules/lr_comparison.png)\n\n"
        f"Selected settings:\n\n```csv\n{table}```\n\n"
        "Use formal_config.json and source_snapshot/ together for the next experiments.\n"
        "Previous cyclic/OneCycle results changed momentum and cannot be pooled with v2.\n"
        "Do not start the three-seed main comparison until this pilot has been reviewed.\n", encoding="utf-8")
    print(f"COMPLETE={root}", flush=True)


if __name__ == "__main__":
    main()
