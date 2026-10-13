import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from an epoch CSV log.")
    parser.add_argument("epoch_csv")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    csv_path = Path(args.epoch_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(csv_path)
    run_name = csv_path.stem.replace("_epochs", "")

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
    output_path = output_dir / f"{run_name}_curves.png"
    fig.savefig(output_path, dpi=160)
    print(f"Wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
