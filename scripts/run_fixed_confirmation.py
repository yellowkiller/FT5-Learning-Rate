"""Shared execution for fixed Constant LR and Warm-up + Cosine confirmation."""
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
SCRIPT = "scripts/run_fixed_confirmation.py"
POLICIES = {
    "constant": {"previous": "seven", "count": 7, "next": "eight", "params": "fixed_lr"},
    "warmup_cosine": {"previous": "six", "count": 6, "next": "seven",
                      "params": "warmup_epochs=7;start_factor=0.01;cosine_epochs=143;eta_min=0.0"},
}


def expected_lr(policy, epoch):
    elapsed = epoch - 1
    if policy == "constant":
        return .1
    if policy != "warmup_cosine":
        raise ValueError(f"Unsupported fixed policy: {policy}")
    return .001 + .099 * elapsed / 7 if elapsed < 7 else .05 * (1 + math.cos(math.pi * (elapsed - 7) / 143))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize(root, parent, policy):
    spec = POLICIES[policy]
    plan = f"docs/{policy}_confirmation_20261103.md"
    comparison = f"{spec['previous']}_policy_confirmation_per_seed.csv"
    if root.exists():
        raise RuntimeError("Refusing to initialize an existing campaign")
    if read(parent / "status.json")["phase"] != "complete":
        raise RuntimeError("Parent campaign is incomplete")
    previous = read(parent / "protocol/protocol.json")
    source = parent / "protocol/source_snapshot"
    for rel, expected in previous["source_sha256"].items():
        if digest(source / rel) != expected:
            raise RuntimeError(f"Parent source hash mismatch: {rel}")
    historical = PROJECT / "results/lr_pilots/20261016_110243_fixedmomentum_split1_seed1/formal_config.json"
    config = read(historical)["schedulers"][policy]
    if config["lr"] != .1 or config["scheduler_params"] != spec["params"]:
        raise RuntimeError("Historical configuration differs from registered plan")
    shutil.copytree(source, root / "protocol/source_snapshot")
    shutil.copy2(__file__, root / "protocol/source_snapshot" / SCRIPT)
    shutil.copy2(PROJECT / plan, root / "protocol/experiment_plan.md")
    shutil.copy2(historical, root / "protocol/historical_formal_config.json")
    shutil.copy2(parent / "summary" / comparison, root / "protocol" / comparison)
    protocol = {**previous, "root": str(root), "parent": str(parent), "max_runs": 3,
                "parallel_jobs": 1, "tuning_seeds": [], "confirmation_seeds": [6, 7, 8],
                "created": datetime.now().astimezone().isoformat(),
                "accuracy_thresholds": [.8, .9, .93, .94], "stability_window": 10,
                "scope": f"Fixed historical {policy}; three paired seeds; no tuning",
                "fixed_policy": policy, "comparison_file": comparison,
                "selection_scope": "No parameter selection; checkpoint selected by validation only",
                "timing_scope": "New runs serial; historical runs shared GPU; no direct timing speedup claim",
                "historical_configuration_sha256": digest(historical),
                "comparison_csv_sha256": digest(parent / "summary" / comparison),
                "plan_sha256": digest(PROJECT / plan),
                "source_sha256": {**previous["source_sha256"], SCRIPT: digest(Path(__file__))}}
    write(root / "protocol/protocol.json", protocol)
    write(root / "03_frozen_configs" / f"{policy}.json", {
        "selected": {"id": f"{policy}_lr0.1", "scheduler": policy, "lr": .1, "options": {}},
        "frozen_at": protocol["created"], "source": str(historical), "scheduler_metadata": config,
        "reason": "Retain prior validation-pilot configuration without further search"})
    with (root / "protocol/environment.txt").open("w") as handle:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=handle, check=True)
    write(root / "status.json", {"phase": "initialized", "completed": 0, "active": {}})


def execute(root):
    protocol = read(root / "protocol/protocol.json")
    policy = protocol["fixed_policy"]
    spec = POLICIES[policy]
    comparison = protocol["comparison_file"]
    snapshot = root / "protocol/source_snapshot"
    for rel, expected in protocol["source_sha256"].items():
        if digest(snapshot / rel) != expected:
            raise RuntimeError(f"Frozen source changed: {rel}")
    if digest(root / "protocol/experiment_plan.md") != protocol["plan_sha256"]:
        raise RuntimeError("Frozen plan changed")
    if digest(root / "protocol/historical_formal_config.json") != protocol["historical_configuration_sha256"]:
        raise RuntimeError("Historical configuration copy changed")
    if digest(root / "protocol" / comparison) != protocol["comparison_csv_sha256"]:
        raise RuntimeError("Comparison manifest changed")
    sys.path.insert(0, str(snapshot))
    import pandas as pd
    from scripts import run_focused_lr_study as study

    frozen = read(root / "03_frozen_configs" / f"{policy}.json")
    candidate = frozen["selected"]
    expected_candidate = {"id": f"{policy}_lr0.1", "scheduler": policy, "lr": .1, "options": {}}
    if candidate != expected_candidate:
        raise RuntimeError("Frozen candidate changed")
    old = pd.read_csv(root / "protocol" / comparison)
    policies = {"cosine", "warmrestart", "cyclic", "onecycle", "step", "multistep"}
    if spec["count"] == 7:
        policies.add("warmup_cosine")
    if len(old) != spec["count"] * 3 or set(old.scheduler) != policies or old.duplicated(["scheduler", "seed"]).any():
        raise RuntimeError("Invalid parent comparison manifest")
    for name, group in old.groupby("scheduler"):
        if sorted(group.seed.tolist()) != [6, 7, 8]:
            raise RuntimeError(f"Invalid prior confirmation seeds: {name}")
    # Preserve the parent's training source; its legacy preview cannot handle None.
    import torch
    import matplotlib.pyplot as plt
    from schedulers import build_scheduler
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=.1, momentum=.9)
    scheduler, interval, metadata = build_scheduler(policy, optimizer, epochs=150, steps_per_epoch=352)
    if interval != "epoch":
        raise RuntimeError("Unexpected scheduler interval")
    expected_dir = root / "00_implementation_checks/schedules" / candidate["id"]
    expected_dir.mkdir(parents=True, exist_ok=True)
    updates = []
    for epoch in range(1,151):
        for batch in range(1,353):
            updates.append((epoch, batch, optimizer.param_groups[0]["lr"]))
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    pd.DataFrame(updates, columns=["epoch","batch","lr"]).to_csv(expected_dir / "expected_updates.csv", index=False)
    write(expected_dir / "config.json", {**candidate, "metadata": metadata})
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot([v[0] + (v[1]-1)/352 for v in updates], [v[2] for v in updates])
    ax.set(xlabel="Epoch", ylabel="Learning rate", title=candidate["id"])
    fig.tight_layout()
    fig.savefig(expected_dir / "schedule.png", dpi=160)
    plt.close(fig)
    trace = pd.read_csv(root / "00_implementation_checks/schedules" / candidate["id"] / "expected_updates.csv")
    for epoch in range(1, 151):
        expected = expected_lr(policy, epoch)
        actual = trace.loc[trace.epoch == epoch, "lr"]
        if len(actual) != 352 or (actual - expected).abs().max() > 1e-12:
            raise RuntimeError(f"Independent {policy} trace mismatch at epoch{epoch}")
    write(root / "00_implementation_checks/analytic_schedule_check.json", {
        "passed": True, "updates": 52800, "first_lr": float(trace.lr.iloc[0]),
        "scheduler": policy, "last_update_lr": float(trace.lr.iloc[-1])})
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
        combined.to_csv(target / f"{spec['next']}_policy_confirmation_per_seed.csv", index=False)
        combined.groupby("scheduler")[metrics].agg(["mean", "std", "count"]).to_csv(target / f"{spec['next']}_policy_confirmation_summary.csv")
        paired = []
        for row in rows:
            base = next(r for r in previous_rows if r["scheduler"] == "cosine" and r["seed"] == row["seed"])
            paired.append({"seed": row["seed"], **{f"delta_{key}": row[key] - base[key]
                if pd.notna(row[key]) and pd.notna(base[key]) else None for key in metrics}})
        pd.DataFrame(paired).to_csv(target / "paired_vs_cosine.csv", index=False)
        (target / "README.md").write_text(
            f"# Fixed {policy} confirmation\n\nValidated runs: {len(rows)}/3.\n\n"
            f"Historical peak LR=.1; {spec['params']}. No new tuning.\n"
            "Seeds6/7/8; checkpoint selected only by validation accuracy/loss/earlier epoch; one final test evaluation.\n"
            "Tables report per-seed metrics, means and sample SD (ddof=1); count identifies unavailable thresholds.\n"
            "Stability: validation SD over epochs141-150. First80/90/93/94% crossings use validation only; seconds include validation overhead.\n"
            "Comparison tables and paired-vs-Cosine differences are descriptive; three seeds do not establish universal superiority.\n"
            "Prior test results were inspected; this extension is not a pristine independent holdout. No test-driven changes are allowed.\n"
            "New runs are serial; historical runs shared the GPU. Training-time differences cannot establish scheduler speedups.\n"
            "Unequal historical tuning budgets limit fairness claims. The comparison table is complete only after all3 new runs finish.\n", encoding="utf-8")

    report()
    study.state(root, "confirmation", completed=0, total=3)
    for seed in protocol["confirmation_seeds"]:
        row = study.run_one(root, candidate, seed, "04_confirmation", protocol)
        rows.append(enrich(row))
        study.figures(root, [row])
        report()
        study.state(root, "confirmation", completed=len(rows), total=3)
    study.state(root, "complete", completed=3, total=3, summary=str(root / "summary/README.md"))


def main(default_policy=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--policy", choices=tuple(POLICIES), default=default_policy)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.initialize:
        if args.parent is None:
            parser.error("Initialization requires --parent")
        if args.policy is None:
            parser.error("Initialization requires --policy")
        initialize(root, args.parent.resolve(), args.policy)
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


if __name__ == "__main__":
    main()
