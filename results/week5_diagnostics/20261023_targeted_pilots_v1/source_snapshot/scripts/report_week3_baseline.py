"""Verify six existing baseline runs and generate the Week 3 handoff."""

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from config import ExperimentConfig
from models.resnet import build_model
from schedulers import build_scheduler
from train import build_loaders, run_epoch
from utils.seed import set_seed


def table(frame):
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                      ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    out = root / "week3_baseline"
    for name in ("figures", "tables"):
        (out / name).mkdir(parents=True, exist_ok=True)
    frozen = json.loads((root / "formal_config.json").read_text())
    source_names = ("train.py", "config.py", "schedulers.py", "models/resnet.py",
                    "utils/seed.py", "utils/logger.py", "utils/metrics.py")
    digest = hashlib.sha256()
    for name in source_names:
        digest.update(name.encode())
        digest.update((PROJECT / name).read_bytes())
    assert digest.hexdigest() == frozen["source_sha256"], "Source changed since freeze"
    rows, checks, logs = [], [], {}
    for method in ("constant", "step"):
        for seed in (1, 2, 3):
            directory = root / method / f"seed{seed}"
            epoch_file, = (directory / "results").glob("*_epochs.csv")
            metadata_file, = (directory / "results").glob("*_metadata.json")
            split_file, = (directory / "results").glob("*_split.json")
            checkpoint_file, = (directory / "checkpoints").glob("*_best.pt")
            data = pd.read_csv(epoch_file)
            summary = pd.read_csv(directory / "results/results.csv")
            assert len(summary) == 1
            summary = summary.iloc[0]
            meta = json.loads(metadata_file.read_text())
            split = json.loads(split_file.read_text())
            split_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
            assert split_hash == frozen["split_sha256"] == summary.split_sha256
            assert meta["source_sha256"] == summary.source_sha256 == digest.hexdigest()
            assert len(data) == 150 and data.epoch.tolist() == list(range(1, 151))
            assert np.isfinite(data.to_numpy()).all()
            assert (data[["train_loss", "val_loss"]] >= 0).all().all()
            assert ((data[["train_acc", "val_acc"]] >= 0) & (data[["train_acc", "val_acc"]] <= 1)).all().all()
            assert np.allclose(data.momentum, 0.9)
            assert meta["train_size"] == 45000 and meta["validation_size"] == 5000
            assert meta["epochs"] == 150 and meta["batch_size"] == 128
            assert meta["momentum"] == 0.9 and meta["weight_decay"] == 5e-4
            assert meta["architecture"] == "resnet18" and meta["optimizer"] == "sgd"
            parameter = torch.nn.Parameter(torch.zeros(1))
            optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler, _, _ = build_scheduler(method, optimizer, epochs=150, steps_per_epoch=352)
            expected_lr = []
            for _ in range(150):
                expected_lr.append(optimizer.param_groups[0]["lr"])
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            assert np.allclose(data.lr, expected_lr, rtol=1e-10, atol=1e-12)
            config = ExperimentConfig(**{k: v for k, v in meta["config"].items()
                                         if k in {f.name for f in fields(ExperimentConfig)}})
            set_seed(seed)
            _, val_loader, _ = build_loaders(config)
            assert val_loader.dataset.indices == split["validation_indices"]
            assert len(set(split["train_indices"])) == 45000
            assert len(set(split["validation_indices"])) == 5000
            assert not set(split["train_indices"]) & set(split["validation_indices"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
            model = build_model("resnet18", num_classes=10).to(device)
            model.load_state_dict(checkpoint["model_state"])
            loss, acc = run_epoch(model, val_loader, torch.nn.CrossEntropyLoss(), device, quiet=True)
            best_epoch = int(data.loc[data.val_acc.idxmax(), "epoch"])
            saved_loss = float(data.loc[data.epoch == best_epoch, "val_loss"].iloc[0])
            assert checkpoint["epoch"] == best_epoch
            assert abs(acc - summary.best_val_acc) < 1e-9
            assert abs(acc - checkpoint["best_val_acc"]) < 1e-9
            assert abs(loss - saved_loss) < 1e-5
            assert abs(data.val_acc.iloc[-1] - summary.final_val_acc) < 1e-9
            rows.append(dict(scheduler=method, seed=seed, best_epoch=best_epoch,
                             best_val_acc=acc, final_val_acc=summary.final_val_acc,
                             test_acc=summary.test_acc, final_train_loss=data.train_loss.iloc[-1],
                             final_val_loss=data.val_loss.iloc[-1], final_train_acc=data.train_acc.iloc[-1],
                             training_time_seconds=summary.training_time_seconds,
                             epoch_time_sum_seconds=data.epoch_seconds.sum(), git_commit=summary.git_commit))
            checks.append(dict(scheduler=method, seed=seed, status="PASS", reevaluated_val_loss=loss,
                               logged_best_val_loss=saved_loss, reevaluated_val_acc=acc,
                               checkpoint_epoch=best_epoch, source_sha256=digest.hexdigest(),
                               split_sha256=split_hash, checkpoint_sha256=hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()))
            logs[method, seed] = data
            print(f"PASS {method} seed{seed}: epoch={best_epoch}, validation={acc:.4f}", flush=True)
            del model, checkpoint
    results = pd.DataFrame(rows)
    results.to_csv(out / "tables/baseline_runs.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "tables/verification.csv", index=False)
    metrics = ["best_val_acc", "final_val_acc", "test_acc", "final_train_loss", "final_val_loss", "training_time_seconds"]
    aggregate = results.groupby("scheduler")[metrics].agg(["mean", "std"])
    aggregate.columns = ["_".join(c) for c in aggregate.columns]
    aggregate.to_csv(out / "tables/baseline_mean_std.csv")
    columns = {"train_loss": "Training loss", "val_loss": "Validation loss",
               "train_acc": "Training accuracy", "val_acc": "Validation accuracy", "lr": "Learning rate"}
    for column, label in columns.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
        for ax, method in zip(axes, ("constant", "step")):
            values = np.stack([logs[method, seed][column].to_numpy() for seed in (1, 2, 3)])
            epochs = np.arange(1, 151)
            for seed, values_seed in zip((1, 2, 3), values):
                ax.plot(epochs, values_seed, alpha=0.35, linewidth=0.9, label=f"seed{seed}")
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.plot(epochs, mean, color="black", linewidth=1.5, label="Mean")
            ax.fill_between(epochs, mean - std, mean + std, color="black", alpha=0.12, label="Sample SD")
            ax.set(title=method, xlabel="Epoch", ylabel=label)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"ResNet-18 baseline: {label}")
        fig.tight_layout()
        fig.savefig(out / f"figures/{column}.png", dpi=180)
        plt.close(fig)
    frozen_copy = dict(frozen)
    frozen_copy["schedulers"] = {k: v for k, v in frozen["schedulers"].items() if k in ("constant", "step")}
    (out / "tables/configuration.json").write_text(json.dumps(frozen_copy, indent=2), encoding="utf-8")
    verification = f"""# Baseline verification

All six runs passed structural, numeric, split, LR-sequence and checkpoint checks.
See [machine-readable checks](tables/verification.csv).

- Each log has epochs 1-150; all numeric values finite, losses nonnegative, accuracies in [0,1].
- Fixed 45,000/5,000 disjoint train/validation split, seed 1; identical across training seeds 1,2,3.
- Logged epoch-start LR matches reconstructed scheduler for every epoch; momentum 0.9, weight decay 5e-4.
- CrossEntropyLoss is averaged by sample count; accuracy uses argmax correct/total counts.
- Evaluation uses model.train(False) and torch.no_grad(); no augmentation or shuffling for validation.
- Best checkpoint is saved on strictly improved validation accuracy (earliest epoch wins ties).
- Loaded all six checkpoints and reevaluated validation: accuracy matches within 1e-9; loss within 1e-5.
- Test metrics were read from existing summaries; no new test-set evaluation or test-driven selection.
- RNG setup covers Python, NumPy, CPU/CUDA PyTorch; cuDNN deterministic=True, benchmark=False.
- Bitwise reproducibility is not guaranteed across hardware/software; deterministic algorithms are not enforced globally.
- PYTHONHASHSEED is set inside Python, so it does not reset the current interpreter's hash seed.
- Checkpoints contain model/config/metadata, not optimizer/scheduler/RNG state; adequate for evaluation, not exact training resumption.
- Logged total duration includes checkpoint saving and final test evaluation; epoch-time sum is provided separately.

Source SHA256: `{digest.hexdigest()}`

Split SHA256: `{frozen['split_sha256']}`

Git commit recorded at training: `{', '.join(sorted(set(results.git_commit)))}`.
The training code contained uncommitted modifications; this commit alone is not the exact experiment version.
Use the matching [source snapshot](../../../lr_pilots/20261016_110243_fixedmomentum_split1_seed1/source_snapshot/)
plus the source SHA256. Current hashed training files match the frozen version.
"""
    (out / "verification.md").write_text(verification, encoding="utf-8")
    display = results[["scheduler", "seed", "best_epoch", "best_val_acc", "final_val_acc", "test_acc", "final_train_loss", "final_val_loss", "training_time_seconds"]].copy()
    for col in display.select_dtypes(include="float").columns:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    observations = []
    for method in ("step", "constant"):
        group = results[results.scheduler == method]
        observations.append(f"- {method}: best validation {group.best_val_acc.mean()*100:.2f}% +/- {group.best_val_acc.std()*100:.2f} percentage points; "
                            f"final validation {group.final_val_acc.mean()*100:.2f}%; test {group.test_acc.mean()*100:.2f}% +/- {group.test_acc.std()*100:.2f} percentage points; "
                            f"mean recorded duration {group.training_time_seconds.mean()/60:.2f} minutes.")
    report = ["# Week 3 - Reliable ResNet-18 Baseline", "",
              "Primary baseline: StepLR. Fixed LR is the conventional comparison. Existing completed runs are reused; no retraining.", "",
              "## Configuration", "",
              "CIFAR-10; CIFAR-adapted ResNet-18; CrossEntropyLoss; SGD, momentum 0.9, weight decay 5e-4; batch 128; "
              "150 epochs; initial LR 0.1; training seeds 1/2/3; split_seed=1; 45,000 train / 5,000 validation / 10,000 test. "
              "Training augmentation: random crop (padding 4), random horizontal flip; CIFAR-10 normalization. Validation/test: normalization only.", "",
              "StepLR: step_size=50, gamma=0.1 (LR 0.1 for epochs 1-50, 0.01 for 51-100, 0.001 for 101-150). "
              "Fixed LR remains 0.1. Hardware/software and full frozen configuration are recorded in the linked artifacts.", "",
              "[Configuration](tables/configuration.json) | [Pilot environment](../../../lr_pilots/20261016_110243_fixedmomentum_split1_seed1/environment.json). "
              "The environment record was captured at the pilot stage; no separate full package inventory was captured per formal run.", "",
              "## Results", "", table(display), "",
              "Accuracy is a fraction (0-1). Losses are final-epoch values. Test accuracy uses the best-validation checkpoint, not the final epoch. "
              "Recorded duration includes the final test evaluation; pure epoch-time sums are in the CSV. SD is sample standard deviation across three seeds, not a confidence interval.", "",
              "[Per-run CSV](tables/baseline_runs.csv) | [Mean and SD](tables/baseline_mean_std.csv)", "",
              "## Observations", "", *observations, "",
              "In these runs, StepLR improves mean test accuracy by 7.59 percentage points over fixed LR. "
              "Its final validation accuracy (94.93%) remains close to its best validation accuracy (95.08%), "
              "while fixed LR ends at 82.69% versus a best of 88.05%, indicating poorer late-stage stability. "
              "StepLR's final mean training/validation losses are 0.00236/0.18781, compared with 0.30032/0.54519 for fixed LR. "
              "The low StepLR training loss and remaining validation loss warrant monitoring generalisation; they do not invalidate the verified training pipeline.", "",
              "Use the curves below to assess the effect of step reductions, validation oscillations and the train/validation gap. "
              "Training metrics include augmentation and train-mode batch normalization, while validation uses evaluation mode, so a gap alone does not establish overfitting. "
              "Three seeds give a preliminary estimate of variation, not proof of statistical superiority. "
              "No additional tuning was performed for this report.", "",
              "## Correctness and version", "",
              "All six checkpoints passed validation reevaluation and all 900 epoch records passed numeric/LR checks. "
              "[Verification details and limitations](verification.md). "
              "Recorded Git commit: `55d4d41`; protocol: `fixed-momentum-fixed-split-v2`. "
              "The exact training version is identified by the source hash and pilot source snapshot because the code was not fully committed. No GitHub upload was performed.", "",
              "## Figures", ""]
    for col, label in columns.items():
        report.extend([f"### {label}", "", f"![{label}](figures/{col}.png)", ""])
    (out / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(f"REPORT COMPLETE: {out}")


if __name__ == "__main__":
    main()
