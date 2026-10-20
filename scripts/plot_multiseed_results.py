"""Plot existing formal results separately for each training seed."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_training_curves import plot_single_run
from run_smoketest_experiment import save_bar, save_overlay


SCHEDULERS = [
    "constant", "step", "multistep", "cosine", "warmrestart",
    "cyclic", "onecycle", "warmup_cosine",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    records = {}
    for seed in (1, 2, 3):
        logs, summaries, files = {}, [], {}
        for name in SCHEDULERS:
            directory = root / name / f"seed{seed}" / "results"
            epoch_file, = directory.glob("*_epochs.csv")
            data = pd.read_csv(epoch_file)
            summary = pd.read_csv(directory / "results.csv")
            if len(data) != 150 or data.epoch.tolist() != list(range(1, 151)):
                raise ValueError(f"Incomplete epoch log: {epoch_file}")
            if len(summary) != 1 or int(summary.iloc[0].seed) != seed:
                raise ValueError(f"Unexpected summary: {directory}")
            logs[name], files[name] = data, epoch_file
            summaries.append(summary)
        records[seed] = (logs, pd.concat(summaries, ignore_index=True), files)

    index = ["# Formal experiment figures by seed", "",
             "Eight schedulers, 150 epochs each; fixed split_seed=1.", "",
             "LR curves use epoch-log samples, not every optimizer update. "
             "Test accuracy is the final evaluation of the validation-selected checkpoint.", ""]
    for seed, (logs, summary, files) in records.items():
        output = root / "plots" / f"seed{seed}"
        comparison = output / "comparison"
        comparison.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output / "summary.csv", index=False)
        images = []
        for column, label in [
            ("train_loss", "Training loss"), ("val_loss", "Validation loss"),
            ("train_acc", "Training accuracy"), ("val_acc", "Validation accuracy"),
            ("lr", "Learning rate (epoch samples)"),
        ]:
            save_overlay(logs, comparison, column, f"Seed {seed}: {label}", label)
            images.append((label, f"comparison/{column}_comparison.png"))
        for metric, label in [
            ("best_val_acc", "Best validation accuracy"),
            ("final_val_acc", "Final validation accuracy"),
            ("test_acc", "Test accuracy (validation-selected checkpoint)"),
            ("training_time_seconds", "Training time (seconds)"),
        ]:
            save_bar(summary, comparison, metric, f"Seed {seed}: {label}")
            images.append((label, f"comparison/{metric}_bar.png"))
        for name, epoch_file in files.items():
            directory = output / "individual" / name
            plot_single_run(epoch_file, directory)
            image_file = next(directory.glob("*.png"))
            images.append((name, image_file.relative_to(output).as_posix()))
        content = [f"# Seed {seed} figures", "", "[Summary table](summary.csv)", "",
                   "LR curves show epoch-log samples; within-epoch cyclic/OneCycle changes "
                   "are not fully represented. Accuracy values are fractions (0-1).", ""]
        for label, path in images:
            content.extend([f"## {label}", "", f"![{label}]({path})", ""])
        (output / "README.md").write_text("\n".join(content), encoding="utf-8")
        index.append(f"- [Seed {seed}: figures and summary](seed{seed}/README.md)")
        print(f"Seed {seed}: {len(images)} figures -> {output}")
    (root / "plots" / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    plt.close("all")


if __name__ == "__main__":
    main()
