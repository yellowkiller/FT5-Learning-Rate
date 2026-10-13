import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SMOKETEST_SCHEDULERS = [
    "constant",
    "step",
    "multistep",
    "cosine",
    "warmrestart",
    "cyclic",
    "onecycle",
    "warmup_cosine",
]


def slug_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def run_command(command: list[str], cwd: Path) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def plot_single_run(epoch_csv: Path, output_dir: Path) -> None:
    data = pd.read_csv(epoch_csv)
    run_name = epoch_csv.stem.replace("_epochs", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(data["epoch"], data["train_loss"], label="train")
    axes[0].plot(data["epoch"], data["val_loss"], label="validation")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(data["epoch"], data["train_acc"], label="train")
    axes[1].plot(data["epoch"], data["val_acc"], label="validation")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    axes[2].plot(data["epoch"], data["lr"], label="learning rate")
    axes[2].set_ylabel("LR")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.suptitle(run_name)
    fig.tight_layout()
    fig.savefig(output_dir / f"{run_name}_curves.png", dpi=160)
    plt.close(fig)


def collect_results(experiment_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summaries = []
    epoch_logs = {}
    for scheduler_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
        results_dir = scheduler_dir / "results"
        summary_path = results_dir / "results.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        summary["scheduler_dir"] = scheduler_dir.name
        summaries.append(summary)

        epoch_files = sorted(results_dir.glob("*_epochs.csv"))
        if epoch_files:
            epoch_logs[scheduler_dir.name] = pd.read_csv(epoch_files[-1])

    if not summaries:
        raise FileNotFoundError(f"No results found under {experiment_dir}")
    return pd.concat(summaries, ignore_index=True), epoch_logs


def save_bar(summary: pd.DataFrame, output_dir: Path, metric: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    data = summary.sort_values(metric, ascending=False)
    ax.bar(data["scheduler"], data[metric])
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel("Scheduler")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}_bar.png", dpi=160)
    plt.close(fig)


def save_overlay(epoch_logs: dict[str, pd.DataFrame], output_dir: Path, column: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for scheduler, data in epoch_logs.items():
        ax.plot(data["epoch"], data[column], label=scheduler)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{column}_comparison.png", dpi=160)
    plt.close(fig)


def to_markdown_table(data: pd.DataFrame) -> str:
    columns = list(data.columns)
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in data.iterrows():
        values = [str(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_readme(experiment_dir: Path, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    summary_view = summary[
        [
            "scheduler",
            "best_val_acc",
            "final_val_acc",
            "test_acc",
            "training_time_seconds",
            "scheduler_params",
        ]
    ].sort_values("test_acc", ascending=False)
    lines = [
        "# Smoke Test Experiment",
        "",
        "This folder contains the 30-epoch seed-1 smoke test comparison for all scheduler methods mentioned in the project plan.",
        "",
        "## Parameters",
        "",
        f"- Epochs: {args.epochs}",
        f"- Seed: {args.seed}",
        f"- Batch size: {args.batch_size}",
        f"- Initial LR: {args.lr}",
        f"- Device: {args.device}",
        "",
        "## Result Summary",
        "",
        to_markdown_table(summary_view),
        "",
        "## Folder Structure",
        "",
        "- one subfolder per scheduler",
        "- each scheduler folder contains `results/`, `plots/`, and local `checkpoints/`",
        "- comparison tables are in `tables/`",
        "- comparison figures are in `plots/comparison/`",
        "",
    ]
    (experiment_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 30-epoch smoke tests for all scheduler methods.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--root-dir", default="results/smoketests")
    parser.add_argument("--run-name", default="smoketest")
    parser.add_argument("--schedulers", nargs="+", default=SMOKETEST_SCHEDULERS)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scheduler_tag = "-".join(args.schedulers)
    experiment_name = (
        f"{timestamp}_epochs{args.epochs}_seed{args.seed}_batch{args.batch_size}_"
        f"lr{slug_float(args.lr)}_{scheduler_tag}"
    )
    experiment_dir = project_dir / args.root_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    for scheduler in args.schedulers:
        scheduler_dir = experiment_dir / scheduler
        results_dir = scheduler_dir / "results"
        checkpoint_dir = scheduler_dir / "checkpoints"
        plots_dir = scheduler_dir / "plots"
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        run_name = f"{args.run_name}_epochs{args.epochs}"
        train_command = [
            sys.executable,
            "train.py",
            "--data-dir",
            args.data_dir,
            "--run-name",
            run_name,
            "--scheduler",
            scheduler,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
            "--num-workers",
            "0",
            "--device",
            args.device,
            "--output-dir",
            str(results_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--plot-dir",
            str(plots_dir),
            "--quiet",
        ]
        run_command(train_command, project_dir)

        epoch_files = sorted(results_dir.glob("*_epochs.csv"))
        if epoch_files:
            plot_single_run(epoch_files[-1], plots_dir)

    tables_dir = experiment_dir / "tables"
    comparison_plots_dir = experiment_dir / "plots" / "comparison"
    tables_dir.mkdir(parents=True, exist_ok=True)
    comparison_plots_dir.mkdir(parents=True, exist_ok=True)
    summary, epoch_logs = collect_results(experiment_dir)
    summary_path = tables_dir / "smoketest_summary.csv"
    summary.to_csv(summary_path, index=False)

    save_bar(summary, comparison_plots_dir, "best_val_acc", "Best Validation Accuracy by Scheduler")
    save_bar(summary, comparison_plots_dir, "test_acc", "Test Accuracy by Scheduler")
    save_overlay(epoch_logs, comparison_plots_dir, "train_loss", "Training Loss Comparison", "Train loss")
    save_overlay(epoch_logs, comparison_plots_dir, "val_loss", "Validation Loss Comparison", "Validation loss")
    save_overlay(epoch_logs, comparison_plots_dir, "train_acc", "Training Accuracy Comparison", "Train accuracy")
    save_overlay(epoch_logs, comparison_plots_dir, "val_acc", "Validation Accuracy Comparison", "Validation accuracy")
    save_overlay(epoch_logs, comparison_plots_dir, "lr", "Learning-Rate Schedule Comparison", "Learning rate")
    write_readme(experiment_dir, summary, args)

    print(f"Smoke test complete: {experiment_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

