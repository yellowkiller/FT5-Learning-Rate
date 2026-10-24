"""Audit completed runs, then run a bounded validation-only diagnostic campaign."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from config import ExperimentConfig
from models.resnet import build_model
from schedulers import build_scheduler
from train import build_loaders, write_json
from utils.seed import set_seed
from scripts.plot_training_curves import plot_single_run

OLD = "20261017_092528_epochs150_seeds1-3_fixedsplit_v2"
PILOT = "20261016_110243_fixedmomentum_split1_seed1"


def status(root, phase, **extra):
    write_json(root / "status.json", {"phase": phase, "updated": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **extra})
    print(phase, extra, flush=True)


def save_plot(frame, x, ys, path, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    for y in ys:
        ax.plot(frame[x], frame[y], label=y)
    ax.set(xlabel=x, title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def audit(root, original):
    directory = root / "existing_analysis"
    directory.mkdir(parents=True, exist_ok=True)
    records, windows = [], []
    reference_split = set()
    source_hashes = set()
    source = original / "results/formal_multiseed" / OLD
    for path in sorted(source.glob("*/seed*/results/*_epochs.csv")):
        scheduler, seed = path.parents[2].name, int(path.parents[1].name[4:])
        data = pd.read_csv(path)
        if len(data) != 150 or data.epoch.tolist() != list(range(1, 151)):
            raise RuntimeError(f"Incomplete formal log: {path}")
        if not all(math.isfinite(float(x)) for x in data.select_dtypes("number").to_numpy().flat):
            raise RuntimeError(f"Nonfinite formal metrics: {path}")
        metadata = json.loads(next(path.parent.glob("*_metadata.json")).read_text())
        reference_split.add(metadata["split_sha256"])
        source_hashes.add(metadata["source_sha256"])
        for key, expected in {"momentum": 0.9, "weight_decay": 5e-4, "batch_size": 128,
                              "split_seed": 1, "epochs": 150}.items():
            if metadata[key] != expected:
                raise RuntimeError(f"Protocol mismatch: {path}: {key}")
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=metadata["initial_lr"], momentum=0.9)
        schedule, interval, _ = build_scheduler(scheduler, optimizer, epochs=150, steps_per_epoch=352)
        trace = []
        for _ in range(150):
            trace.append(optimizer.param_groups[0]["lr"])
            for _ in range(352 if interval == "batch" else 1):
                optimizer.step()
                if schedule is not None:
                    schedule.step()
        if not all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(trace, data.lr)):
            raise RuntimeError(f"Default schedule changed: {path}")
        row = {"scheduler": scheduler, "seed": seed, "best_val_acc": data.val_acc.max(),
               "final_val_acc": data.val_acc.iloc[-1], "best_final_gap": data.val_acc.max()-data.val_acc.iloc[-1],
               "last10_val_mean": data.val_acc.tail(10).mean(), "last10_val_sd": data.val_acc.tail(10).std(),
               "final_lr": data.lr.iloc[-1], "epoch_seconds_mean": data.epoch_seconds.mean()}
        for threshold in (0.8, 0.9):
            reached = data[data.val_acc >= threshold]
            first = int(reached.epoch.iloc[0]) if len(reached) else None
            row[f"epoch_to_{int(threshold*100)}"] = first
            row[f"seconds_to_{int(threshold*100)}"] = float(data.epoch_seconds.iloc[:first].sum()) if first else None
        records.append(row)
        if scheduler == "warmrestart":
            for restart in (38, 112):
                before = data[(data.epoch >= restart-5) & (data.epoch < restart)]
                after = data[(data.epoch >= restart) & (data.epoch < restart+5)]
                windows.append({"seed": seed, "restart_epoch": restart,
                                "before_val": before.val_acc.mean(), "after_val": after.val_acc.mean(),
                                "before_lr": before.lr.mean(), "after_lr": after.lr.mean()})
        if scheduler in ("warmrestart", "cyclic"):
            target = directory / f"{scheduler}_seed{seed}"
            target.mkdir(exist_ok=True)
            for metric in ("lr", "train_loss", "val_loss", "train_acc", "val_acc"):
                save_plot(data, "epoch", [metric], target / f"{metric}.png", f"Existing {scheduler}, seed {seed}")
    frame = pd.DataFrame(records)
    if len(frame) != 24 or len(reference_split) != 1 or len(source_hashes) != 1:
        raise RuntimeError("Expected 24 runs with identical source and split")
    frame.to_csv(directory / "per_seed_convergence.csv", index=False)
    frame.groupby("scheduler").mean(numeric_only=True).to_csv(directory / "scheduler_convergence_mean.csv")
    pd.DataFrame(windows).to_csv(directory / "restart_windows.csv", index=False)
    old_range = pd.read_csv(original / "results/lr_pilots" / PILOT / "lr_range.csv")
    write_json(directory / "audit.json", {"runs": 24, "epoch_lr_matches": 3600,
        "split_sha256": next(iter(reference_split)), "historical_source_sha256": next(iter(source_hashes)),
        "old_range_minimum_lr": float(old_range.loc[old_range.smoothed_loss.idxmin(), "lr"]),
        "old_range_maximum_lr": float(old_range.lr.max()),
        "interpretation": "End-of-scan minimum is not a demonstrated divergence boundary; LR floor and restart windows are associations, not causal proof."})
    (directory / "README.md").write_text(
        "# Existing experiment audit\n\n24 runs; 3600 epoch-start LR samples verified against unchanged defaults.\n\n"
        "- per_seed_convergence.csv: first 80/90% validation crossing (blank = not reached), time, final gaps and last-10 variability.\n"
        "- restart_windows.csv: five-epoch windows around restarts at epochs 38 and 112.\n"
        "- cyclic: triangular2 amplitude halves each cycle; the nonzero LR floor remains.\n"
        "- warmrestart: epoch 150 falls within a high-LR portion of the last cycle.\n"
        "- Neither observation alone establishes causality. New paired diagnostics are required.\n"
        "- Elapsed times are descriptive because historical hardware load was uncontrolled.\n", encoding="utf-8")


def range_test(root, data_dir):
    directory = root / "lr_range"
    directory.mkdir(exist_ok=True)
    set_seed(1)
    config = ExperimentConfig(data_dir=data_dir, seed=1, split_seed=1, device="cuda")
    loader, _, _ = build_loaders(config)
    model = build_model("resnet18", num_classes=10).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)
    model.train()
    iterator = iter(loader)
    smooth, best, rows, reason = 0., float("inf"), [], "scan_limit"
    for step in range(1056):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, targets = next(iterator)
        lr = 1e-5 * (3 / 1e-5) ** (step / 1055)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(inputs.cuda()), targets.cuda())
        value = loss.item()
        if not math.isfinite(value):
            reason = "nonfinite_loss"
            break
        smooth = .98 * smooth + .02 * value
        corrected = smooth / (1 - .98 ** (step+1))
        rows.append({"step": step+1, "lr": lr, "loss": value, "smoothed_loss": corrected})
        if step > 100 and corrected > 4 * best:
            reason = "loss_above_4x_best"
            break
        best = min(best, corrected)
        loss.backward()
        optimizer.step()
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / "range.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(frame.lr, frame.loss, alpha=.25, label="raw training loss")
    ax.semilogx(frame.lr, frame.smoothed_loss, label="smoothed training loss")
    ax.axvline(.1, linestyle="--", color="black", label="historical max LR 0.1")
    ax.set(xlabel="Learning rate", ylabel="Training loss", title="Extended LR range test; fresh seed 1 initialization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "lr_range.png", dpi=160)
    plt.close(fig)
    near = frame.iloc[(frame.lr-.1).abs().argsort()[:1]]
    supported = len(frame) >= 100 and frame.lr.max() >= .1 and float(near.smoothed_loss.iloc[0]) < 4 * best
    write_json(directory / "assessment.json", {"stop_reason": reason, "steps": len(frame),
        "maximum_observed_lr": float(frame.lr.max()), "minimum_smoothed_loss_lr": float(frame.loc[frame.smoothed_loss.idxmin(), "lr"]),
        "historical_max_0p1_passes_safety_gate": supported,
        "scope": "Training-only screening; minimum is not optimal validation LR. Shared overhead for both batch schedulers. No range-test weights reused."})
    del model, optimizer, loader
    torch.cuda.empty_cache()
    if not supported:
        raise RuntimeError("Historical max LR 0.1 failed screening; review range evidence before pilots")


def schedule_preview(root, candidate, steps):
    directory = root / "schedules" / candidate["id"]
    directory.mkdir(parents=True, exist_ok=True)
    options = candidate["options"]
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=.1, momentum=.9, weight_decay=5e-4)
    scheduler, interval, metadata = build_scheduler(candidate["scheduler"], optimizer,
        epochs=steps, steps_per_epoch=352, options=options)
    rows = []
    count = steps*352 if interval == "batch" else steps
    for i in range(count):
        rows.append({"epoch": i/352+1 if interval == "batch" else i+1,
                     "lr": optimizer.param_groups[0]["lr"], "momentum": optimizer.param_groups[0]["momentum"]})
        optimizer.step()
        scheduler.step()
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / f"epochs{steps}.csv", index=False)
    save_plot(frame.iloc[::max(1, len(frame)//5000)], "epoch", ["lr"], directory / f"epochs{steps}.png",
              f"{candidate['id']}; {steps} epochs")
    write_json(directory / f"epochs{steps}_metadata.json", metadata)


def train_run(root, candidate, seed, stage, epochs):
    directory = root / stage / candidate["id"] / f"seed{seed}"
    if directory.exists():
        raise RuntimeError(f"Refusing to overwrite existing run {directory}")
    directory.mkdir(parents=True)
    status(root, stage, candidate=candidate["id"], seed=seed, epochs=epochs)
    command = [sys.executable, "-u", "train.py", "--scheduler", candidate["scheduler"],
        "--scheduler-options", json.dumps(candidate["options"]), "--epochs", str(epochs), "--lr", "0.1",
        "--seed", str(seed), "--split-seed", "1", "--batch-size", "128", "--num-workers", "0",
        "--device", "cuda", "--data-dir", str(root_protocol(root)["data_dir"]), "--quiet",
        "--run-name", stage, "--output-dir", str(directory / "results"),
        "--checkpoint-dir", str(directory / "checkpoints")]
    if stage != "formal":
        command.append("--validation-only")
    with (directory / "console.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, "PYTHONHASHSEED": str(seed)})
    path = next((directory / "results").glob("*_epochs.csv"))
    frame = pd.read_csv(path)
    if len(frame) != epochs or not all(math.isfinite(float(x)) for x in frame.select_dtypes("number").to_numpy().flat):
        raise RuntimeError(f"Invalid completed run {directory}")
    summary = pd.read_csv(directory / "results/results.csv").iloc[0].to_dict()
    if stage != "formal" and not pd.isna(summary["test_acc"]):
        raise RuntimeError("Pilot unexpectedly evaluated test data")
    if summary["split_sha256"] != root_protocol(root)["split_sha256"]:
        raise RuntimeError("New run split differs from historical partition")
    plot_single_run(path, directory / "plots")
    for metric in ("train_loss", "val_loss", "train_acc", "val_acc", "lr"):
        save_plot(frame, "epoch", [metric], directory / "plots" / f"{metric}.png", f"{candidate['id']}, seed {seed}")
    return {**summary, "candidate_id": candidate["id"], "selection_val_mean": frame.val_acc.tail(5).mean(),
            "selection_val_loss": frame.val_loss.tail(5).mean()}


def root_protocol(root):
    return json.loads((root / "protocol.json").read_text())


def run_parallel(root, candidates, seeds, stage):
    results = []
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ft5-train") as pool:
        futures = [pool.submit(train_run, root, candidate, seed, stage,
                                150 if stage == "formal" else candidate["pilot_epochs"])
                   for candidate in candidates for seed in seeds]
        for future in as_completed(futures):
            results.append(future.result())
            pd.DataFrame(results).to_csv(root / ("pilot_results.csv" if stage == "pilot" else "formal_results.csv"), index=False)
    return results


def finish_comparisons(root, pilot_rows, formal_rows):
    directory = root / "comparison"
    directory.mkdir(exist_ok=True)
    pilots = pd.DataFrame(pilot_rows)
    report = ["# Diagnostic results", "", "Selection uses validation only. Test scores are descriptive after selection.", ""]
    for scheduler, group in pilots.groupby("scheduler"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for row in group.itertuples():
            path = next((root / "pilot" / row.candidate_id / f"seed{int(row.seed)}" / "results").glob("*_epochs.csv"))
            data = pd.read_csv(path)
            for ax, metric in zip(axes, ("val_acc", "val_loss")):
                ax.plot(data.epoch, data[metric], label=f"{row.candidate_id}; seed {int(row.seed)}")
                ax.set(xlabel="Epoch", ylabel=metric)
                ax.legend(fontsize=7)
        fig.suptitle(f"Paired {scheduler} diagnostic")
        fig.tight_layout()
        fig.savefig(directory / f"{scheduler}_pilot.png", dpi=160)
        plt.close(fig)
        report.extend([f"## {scheduler}", "", f"![Pilot curves]({scheduler}_pilot.png)", ""])
    historical = Path(root_protocol(root)["original_project"]) / "results/formal_multiseed" / OLD
    original = pd.concat([pd.read_csv(p) for p in historical.glob("*/seed*/results/results.csv")], ignore_index=True)
    original["configuration_group"] = "historical_original"
    all_rows = [original]
    if formal_rows:
        new = pd.DataFrame(formal_rows)
        new["configuration_group"] = "new_selected"
        all_rows.append(new)
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(directory / "formal_per_seed_separate_versions.csv", index=False)
    metrics = ["best_val_acc", "final_val_acc", "test_acc", "training_time_seconds"]
    summary = combined.groupby(["scheduler", "configuration_group"])[metrics].agg(["mean", "std", "count"])
    summary.to_csv(directory / "formal_summary_separate_versions.csv")
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 5))
        labels = [f"{name}\n{version}" for name, version in summary.index]
        ax.bar(range(len(labels)), summary[(metric, "mean")], yerr=summary[(metric, "std")], capsize=3)
        ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=8)
        ax.set(ylabel=metric, title="Formal results: configurations remain separate; n=3 each")
        fig.tight_layout()
        fig.savefig(directory / f"{metric}.png", dpi=160)
        plt.close(fig)
    report.extend(["## Formal comparison", "", "![Best validation](best_val_acc.png)", "",
        "![Final validation](final_val_acc.png)", "", "![Test accuracy](test_acc.png)", "",
        "![Historical timing is descriptive](training_time_seconds.png)", "",
        "Tables retain historical and new configuration groups separately. Do not pool them or infer significance from three runs."])
    (directory / "README.md").write_text("\n".join(report), encoding="utf-8")


def execute(root):
    protocol = root_protocol(root)
    expected = json.loads((root / "source_sha256.json").read_text())
    for rel, digest in expected.items():
        if hashlib.sha256((PROJECT / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen source changed: {rel}")
    status(root, "lr_range")
    if not (root / "lr_range/assessment.json").is_file():
        range_test(root, protocol["data_dir"])
    candidates = protocol["candidates"]
    rows = pd.read_csv(root / "pilot_results.csv").to_dict("records") if (root / "pilot_results.csv").is_file() else []
    selected = []
    for scheduler in ("cyclic", "onecycle", "warmrestart"):
        group = [c for c in candidates if c["scheduler"] == scheduler]
        pending = [candidate for candidate in group if not any(
            r["candidate_id"] == candidate["id"] and int(r["seed"]) == 1 for r in rows)]
        if pending:
            rows.extend(run_parallel(root, pending, [1], "pilot"))
            pd.DataFrame(rows).to_csv(root / "pilot_results.csv", index=False)
        scores = {r["candidate_id"]: r["selection_val_mean"] for r in rows if r["scheduler"] == scheduler and r["seed"] == 1}
        delta = scores[group[1]["id"]]-scores[group[0]["id"]]
        changed = delta > .002
        if abs(delta) <= .002:
            pending = [candidate for candidate in group if not any(
                r["candidate_id"] == candidate["id"] and int(r["seed"]) == 2 for r in rows)]
            if pending:
                rows.extend(run_parallel(root, pending, [2], "pilot"))
                pd.DataFrame(rows).to_csv(root / "pilot_results.csv", index=False)
            seed2 = {r["candidate_id"]: r["selection_val_mean"] for r in rows if r["scheduler"] == scheduler and r["seed"] == 2}
            delta2 = seed2[group[1]["id"]]-seed2[group[0]["id"]]
            changed = delta > 0 and delta2 > 0 and (delta+delta2)/2 > .002
        winner = group[1] if changed else group[0]
        selected.append({**winner, "changed_from_original": bool(changed), "seed1_delta": float(delta)})
        write_json(root / "selected_parameters.json", {"selected": selected,
            "rule": "Alternative requires >0.2 percentage point validation tail-5 improvement; close pairs get seed2; require positive paired deltas and mean >0.2pp, otherwise retain original. No test selection."})
    pd.DataFrame(selected).to_csv(root / "selected_parameters.csv", index=False)
    formal = []
    for candidate in selected:
        if candidate["changed_from_original"]:
            formal.extend(run_parallel(root, [{**candidate, "pilot_epochs": 150}], [1, 2, 3], "formal"))
            pd.DataFrame(formal).to_csv(root / "formal_results.csv", index=False)
    if formal:
        pd.DataFrame(formal).groupby("scheduler")[["best_val_acc", "final_val_acc", "test_acc", "training_time_seconds"]].agg(["mean", "std"]).to_csv(root / "formal_summary.csv")
    finish_comparisons(root, rows, formal)
    write_json(root / "completion.json", {"pilot_runs": len(rows), "new_formal_runs": len(formal),
        "unchanged_strategies": "Reuse old parameter results; do not pool configurations.",
        "limitation": "Targeted post-hoc diagnosis after prior test results were visible; tuning uses validation only. New CPU/GPU timing is not automatically comparable to historical timings."})
    with (root / "README.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## Completed\n\nSee selected_parameters.csv, pilot_results.csv and completion.json.\n"
                     "New formal results, if any, remain separate from original runs; formal_summary.csv reports only changed configurations.\n")
    status(root, "complete", pilot_runs=len(rows), new_formal_runs=len(formal))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.execute:
        try:
            execute(root)
        except Exception as error:
            status(root, "failed", error=repr(error))
            raise
        return
    root.mkdir(parents=True, exist_ok=False)
    audit(root, PROJECT)
    protocol = {"original_project": str(PROJECT), "data_dir": str(PROJECT.parent / "data"),
        "epochs_formal": 150, "formal_seeds": [1, 2, 3], "split_seed": 1,
        "split_sha256": json.loads((root / "existing_analysis/audit.json").read_text())["split_sha256"],
        "momentum": .9, "weight_decay": .0005, "batch_size": 128, "num_workers": 0,
        "candidates": [
            {"id": "cyclic_original", "scheduler": "cyclic", "options": {"base_lr": .01}, "pilot_epochs": 30},
            {"id": "cyclic_lower_floor", "scheduler": "cyclic", "options": {"base_lr": .001}, "pilot_epochs": 30},
            {"id": "onecycle_original", "scheduler": "onecycle", "options": {"pct_start": .3}, "pilot_epochs": 30},
            {"id": "onecycle_shorter_rise", "scheduler": "onecycle", "options": {"pct_start": .1}, "pilot_epochs": 30},
            {"id": "warmrestart_original", "scheduler": "warmrestart", "options": {"T_0": 37}, "pilot_epochs": 150},
            {"id": "warmrestart_aligned", "scheduler": "warmrestart", "options": {"T_0": 50}, "pilot_epochs": 150}],
        "tuning_budget": "Targeted diagnostic; two candidates each; 30 epochs for batch schedules; 150 for restart comparison to cover original restarts. Unequal across families, equal within each pair. Extra seed2 only for close pairs.",
        "selection": "Validation-only last-five-epoch mean; alternative needs >0.2pp improvement; close pairs get seed2; retain original if inconclusive.",
        "rationale": {"cyclic": "10x lower floor tests late LR-floor limitation, max and cycle length fixed",
                      "onecycle": "10% vs 30% rise tests schedule allocation, max fixed",
                      "warmrestart": "50+100 epoch periods finish decay near budget end; compare with 37+74 then unfinished 148"}}
    write_json(root / "protocol.json", protocol)
    for candidate in protocol["candidates"]:
        for epochs in sorted({candidate["pilot_epochs"], 150}):
            schedule_preview(root, candidate, epochs)
    frozen = root / "source_snapshot"
    hashes = {}
    files = list(PROJECT.glob("*.py"))
    for directory in ("models", "utils", "scripts", "tests"):
        files.extend((PROJECT / directory).rglob("*.py"))
    for path in files:
        rel = path.relative_to(PROJECT)
        target = frozen / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(root / "source_sha256.json", hashes)
    write_json(root / "environment.json", {"python": sys.version, "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None})
    (root / "README.md").write_text(
        "# Week 5 diagnostics and conditional Week 6-7 follow-up\n\n"
        "## Scope\n\nExisting 24 formal runs are preserved. This is targeted exploratory diagnosis, not equal-budget retuning of all eight strategies. Prior test scores were visible; new choices use validation only.\n\n"
        "## Order\n\n1. Existing log audit and convergence analysis (existing_analysis/).\n"
        "2. Extended training-only range test (lr_range/); halt if max LR 0.1 fails safety gate.\n"
        "3. Paired pilots (pilot/): cyclic floor, OneCycle rise fraction, restart length. Fresh state for every run.\n"
        "4. Select using validation tail-5 mean; retain defaults if inconclusive.\n"
        "5. Only changed strategies receive new 150-epoch runs on seeds 1/2/3 (formal/). Never pool parameter versions.\n\n"
        "## Budget and caveats\n\n420 pilot epochs initially; up to 420 additional epochs for close-pair seed2 checks; up to 1350 formal epochs. GPU runtime can span many hours.\n"
        "Cyclic pilots retain the formal batch cycle length. OneCycle pilots compress the schedule to 30 epochs, so extrapolation is uncertain. Restart pilots use the full 150 epochs to cover the relevant restart windows.\n"
        "LR range screening provides safety evidence, not optimal hyperparameters. Threshold 0.2pp is a practical decision rule, not a significance test.\n\n"
        "## Files\n\nprotocol.json freezes candidates and selection rules. schedules/ contains full traces and pre-training plots. source_snapshot/ and source_sha256.json preserve the runnable code. status.json reports progress/failure. Each run retains logs, metadata, checkpoint and separate plots.\n",
        encoding="utf-8")
    status(root, "prepared", frozen_source=str(frozen))


if __name__ == "__main__":
    main()
