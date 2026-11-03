"""Run bounded validation follow-ups and two staged-decay policies after parent completion."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_focused_lr_study as study
from train import write_json


def audit_parent(parent):
    protocol = json.loads((parent / "protocol/protocol.json").read_text())
    for rel, digest in protocol["source_sha256"].items():
        if hashlib.sha256((parent / "protocol/source_snapshot" / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Parent source changed: {rel}")
    rows = []
    for path in sorted(parent.glob("0[124]_*/**/candidate.json")):
        directory = path.parent
        c = json.loads(path.read_text())
        stage = directory.relative_to(parent).parts[0]
        seed = int(directory.name.removeprefix("seed"))
        if not (directory / "complete.json").exists():
            raise RuntimeError(f"Parent run not validated: {directory}")
        row = study.validate(directory, c, seed, stage, protocol)
        if len(list((directory / "plots").glob("*.png"))) < 6:
            raise RuntimeError(f"Parent run plots missing: {directory}")
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(rows) != 35 or frame.groupby("stage").size().to_dict() != {
            "01_tuning": 15, "02_replication": 8, "04_confirmation": 12}:
        raise RuntimeError("Parent campaign incomplete")
    frozen = list((parent / "03_frozen_configs").glob("*.json"))
    if len(frozen) != 4:
        raise RuntimeError("Parent frozen configurations missing")
    first_test = min(pd.Timestamp(r["date"]) for r in rows if r["stage"] == "04_confirmation")
    # Frozen files were created before confirmation; additionally retain validation-only selection evidence.
    if any(pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC") > first_test for p in frozen):
        raise RuntimeError("Parent freeze timestamp is later than confirmation start")
    return protocol, frame


def prepare(root, parent, parent_protocol):
    if (root / "protocol/protocol.json").exists():
        return json.loads((root / "protocol/protocol.json").read_text())
    if root.exists():
        raise RuntimeError("Unexpected partial follow-up setup; inspect before reuse")
    target = root / "protocol/source_snapshot"
    shutil.copytree(parent / "protocol/source_snapshot", target)
    shutil.copy2(__file__, target / "scripts/run_lr_followup.py")
    shutil.copy2(PROJECT / "docs/conditional_lr_followup_20261101.md", root / "protocol/experiment_plan.md")
    protocol = {**parent_protocol, "root": str(root), "parent": str(parent), "max_runs": 19,
        "targeted_validation_seed": 9, "tuning_seeds": [4, 5], "confirmation_seeds": [6, 7, 8],
        "scope": "At most 3 validation-only gap checks plus 16 Step/MultiStep runs; no global optimality claim",
        "selection_scope": "Additional four-policy checks do not alter parent frozen configurations",
        "source_sha256": {**parent_protocol["source_sha256"],
            "scripts/run_lr_followup.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    write_json(root / "protocol/protocol.json", protocol)
    write_json(root / "protocol/seed_registry.json", {"tuning": [4, 5], "targeted_only": [9],
        "confirmation": [6, 7, 8], "note": "Reuse confirmation seeds for paired comparisons; none used to select Step/MultiStep settings."})
    return protocol


def report(root, rows, selected, targeted_notes):
    target = root / "summary"
    target.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(target / "all_runs.csv", index=False)
    write_json(target / "selected_parameters.json", selected)
    write_json(target / "targeted_validation_assessment.json", targeted_notes)
    formal = pd.DataFrame([r for r in rows if r["stage"] == "04_confirmation"])
    if len(formal):
        formal.groupby("scheduler")[["best_val_acc", "final_val_acc", "test_acc", "last10_val_sd"]].agg(
            ["mean", "std", "count"]).to_csv(target / "confirmation_summary.csv")
    lines = ["# Conditional LR follow-up", "", f"Validated runs: {len(rows)} / at most 19.", "",
        "## Additional validation evidence", "", *targeted_notes, "", "## New-policy configurations", ""]
    lines += [f"- {name}: {choice['selected']}; uncertain={choice['uncertain']}" for name, choice in selected.items()]
    lines += ["", "## Scope and next steps", "",
        "Follow-up evidence does not rewrite the parent campaign or its frozen choices.",
        "Uncertain differences remain uncertain; no unbounded search is triggered.",
        "Compare all six policies using paired seeds6/7/8; do not pool tuning results.",
        "The official test set has been inspected previously, so it is not a pristine untouched benchmark.",
        "Wall times from shared GPU jobs are descriptive. Next evaluate controlled serial timing and one secondary workload."]
    (target / "README.md").write_text("\n".join(lines), encoding="utf-8")


def execute(parent, root):
    if json.loads((parent / "status.json").read_text())["phase"] != "complete":
        print("WAITING: parent campaign is not complete; no experiments started.", flush=True)
        return
    parent_protocol, old = audit_parent(parent)
    protocol = prepare(root, parent, parent_protocol)
    for rel, digest in protocol["source_sha256"].items():
        if hashlib.sha256((root / "protocol/source_snapshot" / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Follow-up frozen source changed: {rel}")
    rows, selected, notes = [], {}, []
    # This prerequisite is based exclusively on tuning validation results, not test ranking.
    cyclic = old[(old.scheduler == "cyclic") & (old.stage == "02_replication")]
    ids = sorted(cyclic.candidate_id.unique())
    if len(ids) != 2:
        raise RuntimeError("Expected two Cyclic finalists")
    deltas = []
    for seed in (4, 5):
        pair = old[(old.scheduler == "cyclic") & (old.seed == seed) & old.candidate_id.isin(ids)].set_index("candidate_id")
        deltas.append(float(pair.loc[ids[0], "best_val_acc"] - pair.loc[ids[1], "best_val_acc"]))
    if deltas[0] * deltas[1] < 0:
        candidates = [json.loads((Path(old[old.candidate_id == name].iloc[0].directory) / "candidate.json").read_text())
                      for name in ids]
        for c in candidates:
            study.preview(root, c)
        added = study.run_group(root, candidates, 9, "00_targeted_validation", protocol)
        rows.extend(added)
        comparison = {name: [float(old[(old.candidate_id == name) & (old.seed == seed)].iloc[0].best_val_acc)
                            for seed in (4, 5)] + [next(r["best_val_acc"] for r in added if r["candidate_id"] == name)]
                      for name in ids}
        notes.append(f"Cyclic paired validation seeds4/5/9: {comparison}. Report direction and spread; no parent configuration is changed.")
    incumbent = study.candidate("onecycle", "lr0.1", .1, pct_start=.3)
    if not len(old[(old.candidate_id == incumbent["id"]) & (old.seed == 5)]):
        study.preview(root, incumbent)
        added = study.run_group(root, [incumbent], 5, "00_targeted_validation", protocol)
        rows.extend(added)
        comparison = {name: [float(old[(old.candidate_id == name) & (old.seed == 4)].iloc[0].best_val_acc),
            added[0]["best_val_acc"] if name == incumbent["id"] else
            float(old[(old.candidate_id == name) & (old.seed == 5)].iloc[0].best_val_acc)]
            for name in ("onecycle_lr0.1", "onecycle_lr0.03")}
        notes.append(f"OneCycle incumbent comparison, validation seeds4/5: {comparison}. No test scores used for this choice.")
    report(root, rows, selected, notes)
    study.figures(root, rows)
    # Bounded follow-ups close evidence gaps; uncertainty is an acceptable outcome, not a reason for endless tuning.
    write_json(root / "protocol/evidence_gate.json", {"parent_runs_validated": 35,
        "targeted_runs": len(rows), "allow_next_two": True,
        "interpretation": "Adequate for bounded pilot/confirmation claims, not proof of globally optimal settings."})
    for name in ("step", "multistep"):
        candidates = [study.candidate(name, f"lr{lr:g}", lr) for lr in (.03, .1, .2)]
        for c in candidates:
            study.preview(root, c)
        screening = study.run_group(root, candidates, 4, "01_tuning", protocol)
        rows.extend(screening)
        finalists = [next(c for c in candidates if c["id"] == r["candidate_id"])
                     for r in study.rank(screening)[:2]]
        replicated = study.run_group(root, finalists, 5, "02_replication", protocol)
        rows.extend(replicated)
        a, b = [c["id"] for c in finalists]
        deltas = [next(r["best_val_acc"] for r in source if r["candidate_id"] == a) -
                  next(r["best_val_acc"] for r in source if r["candidate_id"] == b)
                  for source in (screening, replicated)]
        uncertain = deltas[0]*deltas[1] <= 0 or abs(sum(deltas)/2) <= .002
        winner = f"{name}_lr0.1" if uncertain and any(c["id"] == f"{name}_lr0.1" for c in finalists) else (
            a if uncertain or sum(deltas) > 0 else b)
        selected[name] = {"selected": next(c for c in finalists if c["id"] == winner),
                          "uncertain": uncertain, "paired_deltas": deltas}
        write_json(root / "03_frozen_configs" / f"{name}.json", selected[name])
        report(root, rows, selected, notes)
        study.figures(root, [r for r in rows if r["scheduler"] == name])
    for seed in (6, 7, 8):
        rows.extend(study.run_group(root, [choice["selected"] for choice in selected.values()], seed,
                                    "04_confirmation", protocol))
        report(root, rows, selected, notes)
    study.figures(root, [r for r in rows if r["stage"] == "04_confirmation"])
    combined = pd.concat([old[old.stage == "04_confirmation"],
                          pd.DataFrame([r for r in rows if r["stage"] == "04_confirmation"])], ignore_index=True)
    combined.to_csv(root / "summary/six_policy_confirmation_per_seed.csv", index=False)
    combined.groupby("scheduler")[["best_val_acc", "final_val_acc", "test_acc", "last10_val_sd"]].agg(
        ["mean", "std", "count"]).to_csv(root / "summary/six_policy_confirmation_summary.csv")
    report(root, rows, selected, notes)
    study.state(root, "complete", completed=len(rows), summary=str(root / "summary/README.md"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args.parent.resolve(), args.root.resolve())
    except Exception as error:
        if args.root.exists():
            study.state(args.root.resolve(), "failed", error=str(error))
        raise
