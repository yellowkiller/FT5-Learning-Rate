import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from schedulers import build_scheduler
from scripts import run_fixed_confirmation as fixed
from scripts import run_focused_lr_study as study


class FixedConfirmationTests(unittest.TestCase):
    def test_analytic_trace_matches_both_complete_schedules(self):
        for policy in fixed.POLICIES:
            with self.subTest(policy=policy):
                opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=.1, momentum=.9)
                scheduler, interval, _ = build_scheduler(policy, opt, epochs=150, steps_per_epoch=352)
                self.assertEqual(interval, "epoch")
                for epoch in range(1, 151):
                    self.assertAlmostEqual(opt.param_groups[0]["lr"], fixed.expected_lr(policy, epoch), places=12)
                    opt.step()
                    if scheduler is not None:
                        scheduler.step()

    def test_initialization_freezes_correct_policy_and_manifest(self):
        for policy, spec in fixed.POLICIES.items():
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as temporary:
                project = Path(temporary)
                parent, root = project / "parent", project / "new"
                source = parent / "protocol/source_snapshot"
                source.mkdir(parents=True)
                (source / "scripts").mkdir()
                (source / "train.py").write_text("# frozen training\n")
                fixed.write(parent / "status.json", {"phase": "complete"})
                fixed.write(parent / "protocol/protocol.json", {"source_sha256": {
                    "train.py": hashlib.sha256((source / "train.py").read_bytes()).hexdigest()}})
                (parent / "summary").mkdir()
                comparison = f"{spec['previous']}_policy_confirmation_per_seed.csv"
                (parent / "summary" / comparison).write_text("scheduler,seed\n")
                historical = project / "results/lr_pilots/20261016_110243_fixedmomentum_split1_seed1/formal_config.json"
                fixed.write(historical, {"schedulers": {policy: {"lr": .1, "scheduler_params": spec["params"]}}})
                plan = project / f"docs/{policy}_confirmation_20261103.md"
                plan.parent.mkdir()
                plan.write_text("Frozen test plan")
                with patch.object(fixed, "PROJECT", project), patch.object(fixed.subprocess, "run"):
                    fixed.initialize(root, parent, policy)
                protocol = json.loads((root / "protocol/protocol.json").read_text())
                self.assertEqual(protocol["fixed_policy"], policy)
                self.assertEqual(protocol["comparison_file"], comparison)
                self.assertEqual(protocol["confirmation_seeds"], [6, 7, 8])
                frozen = json.loads((root / "03_frozen_configs" / f"{policy}.json").read_text())
                self.assertEqual(frozen["selected"]["scheduler"], policy)
                for rel, expected in protocol["source_sha256"].items():
                    self.assertEqual(fixed.digest(root / "protocol/source_snapshot" / rel), expected)
                with self.assertRaisesRegex(RuntimeError, "existing campaign"):
                    fixed.initialize(root, parent, policy)

    def test_constant_preview_supports_no_scheduler_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            study.preview(Path(temporary), study.candidate("constant", "lr0.1"))
            trace = Path(temporary) / "00_implementation_checks/schedules/constant_lr0.1/expected_updates.csv"
            with trace.open() as handle:
                lines = handle.readlines()
            self.assertEqual(len(lines), 52801)
            self.assertTrue(all(line.rstrip().endswith(",0.1") for line in lines[1:]))


if __name__ == "__main__":
    unittest.main()
