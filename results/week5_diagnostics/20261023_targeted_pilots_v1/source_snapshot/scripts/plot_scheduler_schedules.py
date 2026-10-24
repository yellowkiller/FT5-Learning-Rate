"""Create separate publication-style LR schedule plots and one summary plot."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCHEDULERS = ["constant", "step", "multistep", "cosine", "warmrestart", "cyclic", "onecycle", "warmup_cosine"]
DESCRIPTIONS = {
    "constant": "Fixed LR: optimizer stays at 0.1 for all epochs.",
    "step": "StepLR: multiply LR by 0.1 at epochs 50 and 100.",
    "multistep": "MultiStepLR: multiply LR by 0.1 at milestones 75 and 112.",
    "cosine": "CosineAnnealingLR: smooth decay from 0.1 toward 0 over 150 epochs.",
    "warmrestart": "CosineAnnealingWarmRestarts: T_0=37, T_mult=2; restarts after epochs 37 and 111.",
    "cyclic": "CyclicLR: batch-level triangular2 cycles between 0.01 and 0.1; momentum cycling is disabled.",
    "onecycle": "OneCycleLR: batch-level warm-up to 0.1, then annealing to a very small final LR.",
    "warmup_cosine": "Warm-up + cosine: linear warm-up from 0.001 for 7 epochs, then cosine decay.",
}


def read_schedule(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    # Keep batch-level schedules readable while retaining the full CSV as evidence.
    if len(data) > 5000:
        indices = sorted(set(range(0, len(data), max(len(data) // 5000, 1))) | {len(data) - 1})
        data = data.iloc[indices].copy()
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedules = {}
    for name in SCHEDULERS:
        data = read_schedule(args.schedule_dir / f"{name}.csv")
        schedules[name] = data
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(data["epoch"], data["lr"], color="#176e68", linewidth=2)
        ax.set(title=f"{name}: learning rate used before optimizer update",
               xlabel="Epoch", ylabel="Learning rate")
        ax.grid(alpha=0.25)
        fig.text(0.02, 0.01, DESCRIPTIONS[name], fontsize=10)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(args.output_dir / f"{name}.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=False)
    for ax, name in zip(axes.flat, SCHEDULERS):
        data = schedules[name]
        ax.plot(data["epoch"], data["lr"], linewidth=1.4, color="#176e68")
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
    fig.suptitle("150-epoch learning-rate schedules: summary", fontsize=16)
    fig.tight_layout()
    fig.savefig(args.output_dir / "summary_all_schedulers.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {len(SCHEDULERS)} individual plots and one summary to {args.output_dir}")


if __name__ == "__main__":
    main()
