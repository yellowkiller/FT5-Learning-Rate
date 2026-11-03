"""Read-only integrity audit and portable catalog; never train or evaluate test images."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "20261101_four_strategies_v1": 35,
    "20261101_conditional_followup_v1": 19,
    "20261103_warmup_cosine_confirmation_v1": 3,
    "20261103_constant_confirmation_v1": 3,
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(project, output):
    base = project / "results/focused_lr_study"
    rows, inventory = [], []
    for name, count in EXPECTED.items():
        root = base / name
        assert read(root / "status.json")["phase"] == "complete", name
        protocol = read(root / "protocol/protocol.json")
        for rel, expected in protocol["source_sha256"].items():
            assert sha256(root / "protocol/source_snapshot" / rel) == expected, rel
        candidates = sorted(root.glob("0[0124]_*/**/candidate.json"))
        assert len(candidates) == count, (name, len(candidates))
        for path in candidates:
            directory = path.parent
            candidate = read(path)
            stage = directory.relative_to(root).parts[0]
            seed = int(directory.name.removeprefix("seed"))
            assert read(directory / "complete.json")["validated"], str(directory)
            epochs = pd.read_csv(next((directory / "results").glob("*_epochs.csv")))
            assert epochs.epoch.tolist() == list(range(1, 151)), str(directory)
            assert epochs.select_dtypes("number").notna().all().all()
            meta = read(next((directory / "results").glob("*_metadata.json")))
            for key, value in {"split_sha256": protocol["split_sha256"], "seed": seed,
                               "epochs": 150, "momentum": .9, "weight_decay": .0005,
                               "batch_size": 128, "architecture": "resnet18", "dataset": "cifar10"}.items():
                assert meta[key] == value, (directory, key)
            split = read(next((directory / "results").glob("*_split.json")))
            actual_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
            assert actual_hash == protocol["split_sha256"]
            summary = pd.read_csv(directory / "results/results.csv")
            assert len(summary) == 1
            row = summary.iloc[0].to_dict()
            assert pd.notna(row["test_acc"]) == (stage == "04_confirmation"), str(directory)
            updates = pd.read_csv(next((directory / "results").glob("*_lr_updates.csv")))
            expected = pd.read_csv(root / "00_implementation_checks/schedules" / candidate["id"] / "expected_updates.csv")
            assert len(updates) == len(expected) == 52800
            assert updates["update"].tolist() == list(range(1, 52801))
            assert (updates.lr - expected.lr).abs().max() < 1e-10
            assert (updates.momentum == .9).all()
            assert updates.epoch.tolist() == expected.epoch.tolist()
            assert updates.batch.tolist() == expected.batch.tolist()
            best = epochs.sort_values(["val_acc", "val_loss", "epoch"], ascending=[False, True, True]).iloc[0]
            best_path = next((directory / "checkpoints").glob("*_best.pt"))
            last_path = next((directory / "checkpoints").glob("*_last.pt"))
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            assert checkpoint["epoch"] == best.epoch
            del checkpoint
            checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
            assert checkpoint["epoch"] == 150 and checkpoint["optimizer_state"]
            del checkpoint
            assert abs(row["best_val_acc"] - best.val_acc) < 1e-12
            assert abs(row["final_val_acc"] - epochs.val_acc.iloc[-1]) < 1e-12
            assert len(list((directory / "plots").glob("*.png"))) >= 6
            portable = {key: value for key, value in row.items() if not key.endswith("_path")}
            portable.update(campaign=name, stage=stage, candidate_id=candidate["id"], seed=seed,
                            directory=directory.relative_to(project).as_posix(),
                            last10_val_sd=epochs.val_acc.tail(10).std())
            for threshold in (.8, .9, .93, .94):
                crossing = epochs[epochs.val_acc >= threshold]
                epoch = int(crossing.epoch.iloc[0]) if len(crossing) else None
                portable[f"epoch_to_{threshold:g}"] = epoch
                portable[f"seconds_to_{threshold:g}"] = float(epochs.epoch_seconds.iloc[:epoch].sum()) if epoch else None
            rows.append(portable)
        print(f"Validated {name}: {count} runs", flush=True)
        # Every original artifact is fingerprinted, including local-only checkpoints/logs.
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".lock"}:
                inventory.append({"path": path.relative_to(project).as_posix(), "bytes": path.stat().st_size,
                                  "sha256": sha256(path), "storage": "local-only" if path.suffix in {".pt", ".log"}
                                  or path.name == "process.json" else "git"})
    frame = pd.DataFrame(rows)
    formal = frame[frame.stage == "04_confirmation"].copy()
    assert len(formal) == 24 and not formal.duplicated(["scheduler", "seed"]).any()
    assert formal.groupby("scheduler").seed.apply(lambda values: sorted(values) == [6, 7, 8]).all()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "all_runs.csv", index=False)
    formal.to_csv(output / "confirmation_per_seed.csv", index=False)
    metrics = ["best_val_acc", "final_val_acc", "test_acc", "last10_val_sd", "training_time_seconds"]
    metrics += [f"{prefix}_to_{threshold:g}" for threshold in (.8, .9, .93, .94) for prefix in ("epoch", "seconds")]
    aggregate = formal.groupby("scheduler")[metrics].agg(["mean", "std", "count"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate.to_csv(output / "confirmation_summary.csv")
    pd.DataFrame(inventory).to_csv(output / "artifact_manifest.csv", index=False)
    (output / "audit.json").write_text(json.dumps({"passed": True, "campaigns": EXPECTED,
        "runs": len(rows), "confirmation_runs": len(formal), "epochs_per_run": 150,
        "checks": ["frozen_source_hashes", "split_hashes", "epoch_rows", "optimizer_update_LRs",
                   "validation_only_tuning", "best_and_last_checkpoints", "plots"],
        "scope": "File integrity audit only; no new training or test-set evaluation",
        "limitations": "Unequal historical tuning budgets; previously inspected test set; mixed serial/shared-GPU timing"}, indent=2), encoding="utf-8")
    print(f"Audit passed: {len(rows)} total runs, {len(formal)} confirmation runs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    audit(project, args.output or project / "results/catalog")
