"""Bounded four-policy search, paired replication and held-out-seed confirmation.

Threads supervise isolated training subprocesses; model/RNG state is never shared.
All candidate decisions use validation only. Completed runs are validated and reused.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from schedulers import build_scheduler
from train import write_json

ACTIVE = {}
LOCK = threading.Lock()


def candidate(scheduler, tag, lr=.1, **options):
    return {"id": f"{scheduler}_{tag}", "scheduler": scheduler, "lr": lr, "options": options}


def first_candidates(name):
    if name in ("cosine", "onecycle"):
        return [candidate(name, f"lr{lr:g}", lr, **({"pct_start": .3} if name == "onecycle" else {}))
                for lr in (.03, .1, .2)]
    if name == "cyclic":
        return [candidate(name, f"floor{floor:g}", base_lr=floor, step_size_up=704)
                for floor in (.01, .001, .0001)]
    return [candidate(name, f"T0_{t}", T_0=t, T_mult=2) for t in (25, 37, 50)]


def extension(best):
    options = dict(best["options"])
    if best["scheduler"] == "onecycle":
        options["pct_start"] = .1
    elif best["scheduler"] == "cyclic":
        options["step_size_up"] = 1760
    else:
        options["T_mult"] = 1
    return candidate(best["scheduler"], best["id"].split("_", 1)[1] + "_extension", best["lr"], **options)


def state(root, phase, **extra):
    with LOCK:
        write_json(root / "status.json", {"phase": phase, "updated": datetime.now().astimezone().isoformat(),
                   "active": dict(ACTIVE), **extra})
    print(phase, extra, flush=True)


def preview(root, c):
    directory = root / "00_implementation_checks/schedules" / c["id"]
    directory.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=c["lr"], momentum=.9)
    sched, interval, metadata = build_scheduler(c["scheduler"], opt, epochs=150,
                                               steps_per_epoch=352, options=c["options"])
    trace = []
    for epoch in range(1, 151):
        for batch in range(1, 353):
            lr = opt.param_groups[0]["lr"]
            if not math.isfinite(lr) or lr < 0 or opt.param_groups[0]["momentum"] != .9:
                raise RuntimeError(f"Invalid schedule {c}")
            trace.append((epoch, batch, lr))
            opt.step()
            if interval == "batch":
                sched.step()
        if interval == "epoch" and sched is not None:
            sched.step()
    pd.DataFrame(trace, columns=["epoch", "batch", "lr"]).to_csv(directory / "expected_updates.csv", index=False)
    write_json(directory / "config.json", {**c, "metadata": metadata})
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([i / 352 for i in range(len(trace))], [v[2] for v in trace])
    ax.set(xlabel="Epoch (zero-based elapsed)", ylabel="LR", title=c["id"])
    fig.tight_layout()
    fig.savefig(directory / "schedule.png", dpi=160)
    plt.close(fig)


def validate(directory, c, seed, stage, protocol):
    logs = list((directory / "results").glob("*_epochs.csv"))
    if len(logs) != 1:
        raise RuntimeError(f"Missing/ambiguous epoch log: {directory}")
    df = pd.read_csv(logs[0])
    if df.epoch.tolist() != list(range(1, 151)) or not all(
            math.isfinite(float(v)) for v in df.select_dtypes("number").to_numpy().flat):
        raise RuntimeError(f"Invalid epochs or metrics: {directory}")
    row = pd.read_csv(directory / "results/results.csv").iloc[0].to_dict()
    meta = json.loads(next((directory / "results").glob("*_metadata.json")).read_text())
    for key, expected in {"split_sha256": protocol["split_sha256"], "seed": seed, "epochs": 150,
                          "momentum": .9, "weight_decay": .0005, "batch_size": 128}.items():
        if meta[key] != expected:
            raise RuntimeError(f"Protocol mismatch {key}: {directory}")
    if stage != "04_confirmation" and not pd.isna(row["test_acc"]):
        raise RuntimeError("Test leakage during tuning")
    if stage == "04_confirmation" and pd.isna(row["test_acc"]):
        raise RuntimeError("Missing confirmation test evaluation")
    lr = pd.read_csv(next((directory / "results").glob("*_lr_updates.csv")))
    expected = pd.read_csv(Path(protocol["root"]) / "00_implementation_checks/schedules" / c["id"] / "expected_updates.csv")
    if len(lr) != 52800 or lr["update"].tolist() != list(range(1, 52801)) or not (
            (lr.lr - expected.lr).abs().max() < 1e-10 and (lr.momentum == .9).all()):
        raise RuntimeError(f"Actual update LR does not match schedule: {directory}")
    best = df.sort_values(["val_acc", "val_loss", "epoch"], ascending=[False, True, True]).iloc[0]
    checkpoint = torch.load(next((directory / "checkpoints").glob("*_best.pt")), map_location="cpu", weights_only=False)
    last = torch.load(next((directory / "checkpoints").glob("*_last.pt")), map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != best.epoch or last["epoch"] != 150 or not last["optimizer_state"]:
        raise RuntimeError(f"Checkpoint mismatch: {directory}")
    row.update(candidate_id=c["id"], stage=stage, last10_val_mean=float(df.val_acc.tail(10).mean()),
               last10_val_sd=float(df.val_acc.tail(10).std()), directory=str(directory), seed=seed)
    for threshold in protocol["accuracy_thresholds"]:
        reached = df[df.val_acc >= threshold]
        epoch = int(reached.epoch.iloc[0]) if len(reached) else None
        row[f"epoch_to_{threshold:g}"] = epoch
        row[f"seconds_to_{threshold:g}"] = float(df.epoch_seconds.iloc[:epoch].sum()) if epoch else None
    return row


def run_one(root, c, seed, stage, protocol):
    directory = root / stage / c["scheduler"] / c["id"] / f"seed{seed}"
    if (directory / "complete.json").exists():
        return validate(directory, c, seed, stage, protocol)
    directory.mkdir(parents=True, exist_ok=True)
    # Never erase partial evidence. An interrupted invocation needs explicit inspection.
    if (directory / "console.log").exists():
        raise RuntimeError(f"Partial run exists; inspect before retry: {directory}")
    write_json(directory / "candidate.json", c)
    cmd = [sys.executable, "-u", "train.py", "--scheduler", c["scheduler"], "--lr", str(c["lr"]),
           "--scheduler-options", json.dumps(c["options"]), "--epochs", "150", "--seed", str(seed),
           "--split-seed", "1", "--batch-size", "128", "--num-workers", "0", "--device", "cuda",
           "--data-dir", protocol["data_dir"], "--quiet", "--run-name", c["id"],
           "--output-dir", str(directory / "results"), "--checkpoint-dir", str(directory / "checkpoints")]
    if stage != "04_confirmation":
        cmd.append("--validation-only")
    with (directory / "console.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=root / "protocol/source_snapshot", stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONHASHSEED": str(seed), "FT5_CPU_THREADS": "4", "OMP_NUM_THREADS": "4"})
        with LOCK:
            ACTIVE[str(proc.pid)] = {"candidate": c["id"], "seed": seed, "stage": stage}
        state(root, stage)
        code = proc.wait()
        with LOCK:
            ACTIVE.pop(str(proc.pid), None)
    if code:
        write_json(directory / "failed.json", {"returncode": code})
        raise RuntimeError(f"Training failed ({code}): {directory / 'console.log'}")
    row = validate(directory, c, seed, stage, protocol)
    write_json(directory / "complete.json", {"validated": True, "epochs": 150})
    state(root, stage, last_completed=c["id"], seed=seed)
    return row


def run_group(root, candidates, seed, stage, protocol):
    rows = []
    with ThreadPoolExecutor(max_workers=protocol["parallel_jobs"]) as pool:
        futures = [pool.submit(run_one, root, c, seed, stage, protocol) for c in candidates]
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def rank(rows):
    return sorted(rows, key=lambda r: (-r["best_val_acc"], r["best_val_loss"], r["best_epoch"], r["candidate_id"]))


def figures(root, rows):
    for row in rows:
        directory = Path(row["directory"])
        df = pd.read_csv(next((directory / "results").glob("*_epochs.csv")))
        target = directory / "plots"
        target.mkdir(exist_ok=True)
        for metric in ("train_loss", "val_loss", "train_acc", "val_acc", "lr"):
            fig, ax = plt.subplots(figsize=(8, 4))
            if metric == "lr":
                updates = pd.read_csv(next((directory / "results").glob("*_lr_updates.csv")))
                ax.plot(updates.epoch + (updates.batch - 1) / 352, updates.lr)
            else:
                ax.plot(df.epoch, df[metric])
            ax.set(xlabel="Epoch", ylabel=metric, title=f"{row['candidate_id']} / seed {row['seed']}")
            fig.tight_layout()
            fig.savefig(target / f"{metric}.png", dpi=140)
            plt.close(fig)
        updates = pd.read_csv(next((directory / "results").glob("*_lr_updates.csv")))
        tail = updates[updates.epoch >= 121]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(tail.epoch + (tail.batch-1)/352, tail.lr.clip(lower=1e-12))
        ax.set(xlabel="Epoch", ylabel="LR (log scale)", title=f"Late LR: {row['candidate_id']}")
        fig.tight_layout()
        fig.savefig(target / "late_lr_logscale.png", dpi=140)
        plt.close(fig)
    for scheduler in sorted({r["scheduler"] for r in rows}):
        selected = [r for r in rows if r["scheduler"] == scheduler]
        target = root / "summary/figures/per_strategy" / scheduler
        target.mkdir(parents=True, exist_ok=True)
        for stage in sorted({r["stage"] for r in selected}):
            for metric in ("val_acc", "val_loss", "lr"):
                fig, ax = plt.subplots(figsize=(10, 5))
                for row in selected:
                    if row["stage"] != stage:
                        continue
                    df = pd.read_csv(next((Path(row["directory"]) / "results").glob("*_epochs.csv")))
                    ax.plot(df.epoch, df[metric], label=f"{row['candidate_id']}, seed {row['seed']}", alpha=.8)
                ax.set(xlabel="Epoch", ylabel=metric, title=f"{scheduler}: {stage}")
                ax.legend(fontsize=7)
                fig.tight_layout()
                fig.savefig(target / f"{stage}_{metric}.png", dpi=160)
                plt.close(fig)


def summarize(root, rows, choices):
    target = root / "summary"
    target.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(target / "all_runs.csv", index=False)
    write_json(target / "selected_parameters.json", choices)
    if choices:
        pd.DataFrame([{"scheduler": key, **value["selected"], "uncertain": value["uncertain"]}
                      for key, value in choices.items()]).to_csv(target / "selected_parameters.csv", index=False)
    formal = df[df.stage == "04_confirmation"]
    if len(formal):
        aggregate = formal.groupby("scheduler")[["best_val_acc", "final_val_acc", "test_acc", "last10_val_sd",
                 "training_time_seconds"]].agg(["mean", "std", "count"])
        aggregate.to_csv(target / "confirmation_summary.csv")
        target_fig = target / "figures/overall"
        target_fig.mkdir(parents=True, exist_ok=True)
        for metric in ("best_val_acc", "final_val_acc", "test_acc"):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(aggregate.index, aggregate[(metric, "mean")], yerr=aggregate[(metric, "std")].fillna(0), capsize=4)
            ax.set(ylabel=metric, title="Frozen configurations, independent confirmation seeds")
            fig.tight_layout()
            fig.savefig(target_fig / f"{metric}.png", dpi=160)
            plt.close(fig)
    lines = ["# Focused LR study", "", f"Validated completed runs: {len(rows)} / at most 35.", "",
             "## Configuration choices (validation only)", ""]
    for name, choice in choices.items():
        lines.append(f"- {name}: {choice['selected']['id']}; uncertain={choice['uncertain']}; {choice['reason']}")
    lines += ["", "## Interpretation and next steps", "",
        "All bounds are finite search candidates, not globally optimal learning rates.",
        "Tuning seeds are excluded from confirmation. Do not merge historical configurations into these estimates.",
        "Parallel wall times are descriptive; a scheduler speed claim requires serial timing on controlled hardware.",
        "Three confirmation seeds are descriptive evidence, not proof of a universal ranking.",
        "If confirmation reverses tuning rankings, report uncertainty; do not tune on the inspected test scores.",
        "Next: inspect per-seed consistency and accuracy/stability trade-offs; choose one secondary workload only after this comparison."]
    if len(formal):
        means = formal.groupby("scheduler").test_acc.mean().sort_values(ascending=False)
        lines += ["", "## Descriptive confirmation results", ""]
        lines += [f"- {name}: mean test accuracy {value:.4%}" for name, value in means.items()]
    (target / "README.md").write_text("\n".join(lines), encoding="utf-8")


def execute(root):
    protocol = json.loads((root / "protocol/protocol.json").read_text())
    for rel, digest in protocol["source_sha256"].items():
        if hashlib.sha256((root / "protocol/source_snapshot" / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen source hash mismatch: {rel}")
    rows, choices = [], {}
    for name in ("cosine", "onecycle", "cyclic", "warmrestart"):
        initial = first_candidates(name)
        for c in initial:
            preview(root, c)
        screening = run_group(root, initial, 4, "01_tuning", protocol)
        rows.extend(screening)
        summarize(root, rows, choices)
        if name != "cosine":
            best = next(c for c in initial if c["id"] == rank(screening)[0]["candidate_id"])
            extra = extension(best)
            preview(root, extra)
            screening += run_group(root, [extra], 4, "01_tuning", protocol)
            rows.append(screening[-1])
            initial.append(extra)
        finalists = [next(c for c in initial if c["id"] == r["candidate_id"]) for r in rank(screening)[:2]]
        replicated = run_group(root, finalists, 5, "02_replication", protocol)
        rows.extend(replicated)
        paired = {c["id"]: [next(r for r in source if r["candidate_id"] == c["id"])
                            for source in (screening, replicated)] for c in finalists}
        a, b = [c["id"] for c in finalists]
        deltas = [paired[a][i]["best_val_acc"] - paired[b][i]["best_val_acc"] for i in (0, 1)]
        uncertain = not (all(d > 0 for d in deltas) or all(d < 0 for d in deltas)) or abs(sum(deltas)/2) <= .002
        # Conservative fallback prefers an existing incumbent among the finalists.
        incumbent = {"cosine": "cosine_lr0.1", "onecycle": "onecycle_lr0.1",
                     "cyclic": "cyclic_floor0.001", "warmrestart": "warmrestart_T0_50"}[name]
        winner = (incumbent if uncertain and incumbent in paired else a) if uncertain else (a if sum(deltas) > 0 else b)
        chosen = next(c for c in finalists if c["id"] == winner)
        choices[name] = {"selected": chosen, "uncertain": uncertain, "paired_deltas": deltas,
                         "reason": "Paired seed4/5 best-validation accuracy; <=0.2pp mean gap or opposing signs is uncertain; incumbent fallback when available, otherwise seed4 leader."}
        write_json(root / "03_frozen_configs" / f"{name}.json", choices[name])
        summarize(root, rows, choices)
        figures(root, [r for r in rows if r["scheduler"] == name])
        state(root, "strategy_tuning_complete", scheduler=name, completed=len(rows))
    # Freeze ALL choices before evaluating any test data.
    state(root, "confirmation", completed=len(rows))
    for seed in (6, 7, 8):
        rows.extend(run_group(root, [v["selected"] for v in choices.values()], seed, "04_confirmation", protocol))
        summarize(root, rows, choices)
    figures(root, [r for r in rows if r["stage"] == "04_confirmation"])
    summarize(root, rows, choices)
    state(root, "complete", completed=len(rows), summary=str(root / "summary/README.md"))


def initialize(root, parallel_jobs):
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("Refusing to overwrite an existing campaign")
    root.mkdir(parents=True, exist_ok=True)
    used = set()
    for path in (PROJECT / "results").rglob("*_metadata.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if "seed" in metadata:
            used.add(metadata["seed"])
    if used.intersection({4, 5, 6, 7, 8}):
        raise RuntimeError(f"Proposed seeds already used: {used}")
    files = ["train.py", "config.py", "schedulers.py", "models/resnet.py",
             "utils/seed.py", "utils/logger.py", "utils/metrics.py", "scripts/run_focused_lr_study.py"]
    hashes = {}
    for rel in files:
        target = root / "protocol/source_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / rel, target)
        hashes[rel] = hashlib.sha256(target.read_bytes()).hexdigest()
    protocol = {"root": str(root), "data_dir": str(PROJECT.parent / "data"), "parallel_jobs": parallel_jobs,
        "epochs": 150, "steps_per_epoch": 352, "tuning_seeds": [4, 5], "confirmation_seeds": [6, 7, 8],
        "previously_used_seeds": sorted(used), "source_sha256": hashes,
        "split_sha256": "eaeb385c407962ceb057cbe0552bff8071a52d9b2c66a0c7432da30205174ad1",
        "accuracy_thresholds": [.90, .93, .94], "stability_window": 10,
        "threshold_rationale": "Pre-registered CIFAR-10 milestones based on historical validation curves; not universal and no replacement after observing new results.",
        "max_runs": 35, "timing_scope": "Shared-GPU wall time is descriptive only",
        "selection": "Best validation accuracy; lower loss and earlier epoch break ties; paired mean <=0.2pp is practical uncertainty, not a significance test",
        "scope": "Sequential one-factor search, unequal 5/6-run effort across policies; not equal-budget global optimization"}
    write_json(root / "protocol/protocol.json", protocol)
    write_json(root / "protocol/candidate_grid.json", {name: first_candidates(name)
        for name in ("cosine", "onecycle", "cyclic", "warmrestart")})
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=(root / "protocol/environment.txt").open("w"), check=True)
    (root / "README.md").write_text("# Focused four-policy LR study\n\n"
        "Isolated subprocesses; up to two concurrent jobs. All runs 150 epochs.\n"
        "01_tuning: 3 initial candidates per policy, seed4; one conditional single-factor extension except cosine.\n"
        "02_replication: top two complete configurations per policy, seed5.\n"
        "03_frozen_configs: all choices fixed before test evaluation.\n"
        "04_confirmation: seeds6/7/8, selected policies only, test set evaluated once per selected checkpoint.\n"
        "No pooling across tuning/confirmation or historical versions. No GitHub upload.\n"
        "Partial logs are never overwritten. status.json and queue logs report failure/completion.\n"
        "Summary figures remain separated by policy and stage. See protocol/ for exact sources, seeds and settings.\n",
        encoding="utf-8")
    state(root, "initialized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--parallel-jobs", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    if args.initialize:
        initialize(args.root.resolve(), args.parallel_jobs)
    else:
        try:
            execute(args.root.resolve())
        except Exception as error:
            state(args.root.resolve(), "failed", error=str(error))
            raise
