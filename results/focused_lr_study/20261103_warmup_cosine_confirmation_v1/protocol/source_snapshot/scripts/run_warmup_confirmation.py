"""Confirm one historical frozen warm-up configuration; never search on test results."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
PLAN = "docs/warmup_cosine_confirmation_20261103.md"
SCRIPT = "scripts/run_warmup_confirmation.py"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize(root, parent):
    if root.exists():
        raise RuntimeError("Refusing to initialize an existing campaign")
    if read(parent / "status.json")["phase"] != "complete":
        raise RuntimeError("Six-policy parent is incomplete")
    previous = read(parent / "protocol/protocol.json")
    source = parent / "protocol/source_snapshot"
    for rel, expected in previous["source_sha256"].items():
        if digest(source / rel) != expected:
            raise RuntimeError(f"Parent source hash mismatch: {rel}")
    historical = PROJECT / "results/lr_pilots/20261016_110243_fixedmomentum_split1_seed1/formal_config.json"
    config = read(historical)["schedulers"]["warmup_cosine"]
    if config["lr"] != .1 or config["scheduler_params"] != "warmup_epochs=7;start_factor=0.01;cosine_epochs=143;eta_min=0.0":
        raise RuntimeError("Historical warm-up configuration differs from registered plan")
    shutil.copytree(source, root / "protocol/source_snapshot")
    shutil.copy2(__file__, root / "protocol/source_snapshot" / SCRIPT)
    shutil.copy2(PROJECT / PLAN, root / "protocol/experiment_plan.md")
    shutil.copy2(historical, root / "protocol/historical_formal_config.json")
    shutil.copy2(parent / "summary/six_policy_confirmation_per_seed.csv", root / "protocol/six_policy_confirmation_per_seed.csv")
    protocol = {**previous, "root": str(root), "parent": str(parent), "max_runs": 3,
                "parallel_jobs": 1, "tuning_seeds": [], "confirmation_seeds": [6, 7, 8],
                "created": datetime.now().astimezone().isoformat(),
                "accuracy_thresholds": [.8, .9, .93, .94], "stability_window": 10,
                "scope": "Fixed historical Warm-up + Cosine; three paired seeds; no tuning",
                "selection_scope": "No parameter selection; checkpoint selected by validation only",
                "timing_scope": "New runs serial; historical runs shared GPU; no direct timing speedup claim",
                "historical_configuration_sha256": digest(historical),
                "comparison_csv_sha256": digest(parent / "summary/six_policy_confirmation_per_seed.csv"),
                "plan_sha256": digest(PROJECT / PLAN),
                "source_sha256": {**previous["source_sha256"], SCRIPT: digest(Path(__file__))}}
    write(root / "protocol/protocol.json", protocol)
    write(root / "03_frozen_configs/warmup_cosine.json", {
        "selected": {"id": "warmup_cosine_lr0.1", "scheduler": "warmup_cosine", "lr": .1, "options": {}},
        "frozen_at": protocol["created"], "source": str(historical), "scheduler_metadata": config,
        "reason": "Retain prior validation-pilot configuration without further search"})
    with (root / "protocol/environment.txt").open("w") as handle:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=handle, check=True)
    write(root / "status.json", {"phase": "initialized", "completed": 0, "active": {}})


def execute(root):
    protocol = read(root / "protocol/protocol.json")
    snapshot = root / "protocol/source_snapshot"
    for rel, expected in protocol["source_sha256"].items():
        if digest(snapshot / rel) != expected:
            raise RuntimeError(f"Frozen source changed: {rel}")
    if digest(root / "protocol/experiment_plan.md") != protocol["plan_sha256"]:
        raise RuntimeError("Frozen plan changed")
    if digest(root / "protocol/historical_formal_config.json") != protocol["historical_configuration_sha256"]:
        raise RuntimeError("Historical configuration copy changed")
    if digest(root / "protocol/six_policy_confirmation_per_seed.csv") != protocol["comparison_csv_sha256"]:
        raise RuntimeError("Comparison manifest changed")
    sys.path.insert(0, str(snapshot))
    import pandas as pd
    from scripts import run_focused_lr_study as study

    frozen = read(root / "03_frozen_configs/warmup_cosine.json")
    candidate = frozen["selected"]
    expected_candidate = {"id": "warmup_cosine_lr0.1", "scheduler": "warmup_cosine", "lr": .1, "options": {}}
    if candidate != expected_candidate:
        raise RuntimeError("Frozen candidate changed")
    old = pd.read_csv(root / "protocol/six_policy_confirmation_per_seed.csv")
    policies = {"cosine", "warmrestart", "cyclic", "onecycle", "step", "multistep"}
    if len(old) != 18 or set(old.scheduler) != policies or old.duplicated(["scheduler", "seed"]).any():
        raise RuntimeError("Invalid six-policy comparison manifest")
    for name, group in old.groupby("scheduler"):
        if sorted(group.seed.tolist()) != [6, 7, 8]:
            raise RuntimeError(f"Invalid prior confirmation seeds: {name}")
    study.preview(root, candidate)
    trace = pd.read_csv(root / "00_implementation_checks/schedules" / candidate["id"] / "expected_updates.csv")
    for epoch in range(1, 151):
        elapsed = epoch - 1
        expected = .001 + .099 * elapsed / 7 if elapsed < 7 else .05 * (1 + math.cos(math.pi * (elapsed - 7) / 143))
        actual = trace.loc[trace.epoch == epoch, "lr"]
        if len(actual) != 352 or (actual - expected).abs().max() > 1e-12:
            raise RuntimeError(f"Independent warm-up trace mismatch at epoch{epoch}")
    write(root / "00_implementation_checks/analytic_schedule_check.json", {
        "passed": True, "updates": 52800, "first_lr": float(trace.lr.iloc[0]),
        "peak_epoch": 8, "last_update_lr": float(trace.lr.iloc[-1])})
    rows = []
    metrics = ["best_val_acc", "final_val_acc", "test_acc", "last10_val_sd", "training_time_seconds"]
    metrics += [f"{prefix}_to_{threshold:g}" for threshold in protocol["accuracy_thresholds"] for prefix in ("epoch", "seconds")]

    def enrich(row):
        df = pd.read_csv(next((Path(row["directory"]) / "results").glob("*_epochs.csv")))
        if df.epoch.tolist() != list(range(1, 151)) or row["split_sha256"] != protocol["split_sha256"]:
            raise RuntimeError("Comparison run protocol mismatch")
        for threshold in protocol["accuracy_thresholds"]:
            crossed = df[df.val_acc >= threshold]
            epoch = int(crossed.epoch.iloc[0]) if len(crossed) else None
            row[f"epoch_to_{threshold:g}"] = epoch
            row[f"seconds_to_{threshold:g}"] = float(df.epoch_seconds.iloc[:epoch].sum()) if epoch else None
        return row

    previous_rows = [enrich(r) for r in old.to_dict("records")]

    def report():
        target = root / "summary"
        target.mkdir(exist_ok=True)
        new = pd.DataFrame(rows)
        new.to_csv(target / "all_runs.csv", index=False)
        if len(new):
            new.groupby("scheduler")[metrics].agg(["mean", "std", "count"]).to_csv(target / "confirmation_summary.csv")
        combined = pd.DataFrame(previous_rows + rows)
        combined.to_csv(target / "seven_policy_confirmation_per_seed.csv", index=False)
        combined.groupby("scheduler")[metrics].agg(["mean", "std", "count"]).to_csv(target / "seven_policy_confirmation_summary.csv")
        paired = []
        for row in rows:
            base = next(r for r in previous_rows if r["scheduler"] == "cosine" and r["seed"] == row["seed"])
            paired.append({"seed": row["seed"], **{f"delta_{key}": row[key] - base[key]
                if pd.notna(row[key]) and pd.notna(base[key]) else None for key in metrics}})
        pd.DataFrame(paired).to_csv(target / "paired_vs_cosine.csv", index=False)
        (target / "README.md").write_text(
            f"# Fixed Warm-up + Cosine confirmation\n\nValidated runs: {len(rows)}/3.\n\n"
            "Historical LR=.1, warmup7 epochs, start_factor=.01, cosine143 epochs, eta_min=0. No new tuning.\n"
            "Seeds6/7/8; checkpoint selected only by validation accuracy/loss/earlier epoch; one final test evaluation.\n"
            "Tables report per-seed metrics, means and sample SD (ddof=1); count identifies unavailable thresholds.\n"
            "Stability: validation SD over epochs141-150. First80/90/93/94% crossings use validation only; seconds include validation overhead.\n"
            "Seven-policy tables and paired-vs-Cosine differences are descriptive; three seeds do not establish universal superiority.\n"
            "Prior test results were inspected; this extension is not a pristine independent holdout. No test-driven changes are allowed.\n"
            "New runs are serial; historical runs shared the GPU. Training-time differences cannot establish scheduler speedups.\n"
            "Unequal historical tuning budgets limit fairness claims. ConstantLR is outside this campaign.\n", encoding="utf-8")

    report()
    study.state(root, "confirmation", completed=0, total=3)
    for seed in protocol["confirmation_seeds"]:
        row = study.run_one(root, candidate, seed, "04_confirmation", protocol)
        rows.append(enrich(row))
        study.figures(root, [row])
        report()
        study.state(root, "confirmation", completed=len(rows), total=3)
    study.state(root, "complete", completed=3, total=3, summary=str(root / "summary/README.md"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.initialize:
        if args.parent is None:
            parser.error("Initialization requires --parent")
        initialize(root, args.parent.resolve())
    else:
        # A persistent exclusive lock prevents concurrent starts and automatic retries after failure.
        lock = root / "queue.lock"
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        try:
            execute(root)
        except Exception as error:
            write(root / "status.json", {"phase": "failed", "error": str(error),
                "updated": datetime.now().astimezone().isoformat(), "note": "Preserve partial runs; inspect before retry"})
            raise
