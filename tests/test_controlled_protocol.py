import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from config import ExperimentConfig
from schedulers import build_scheduler
from train import build_loaders, parse_args, run_epoch


class FakeCifar:
    def __init__(self, root, train, download, transform):
        self.size = 50000 if train else 10000

    def __len__(self):
        return self.size


class ControlledProtocolTests(unittest.TestCase):
    def test_validation_metrics_weight_batches_and_disable_gradients(self):
        class FixedModel(torch.nn.Module):
            def forward(self, inputs):
                self.assert_eval = not self.training and not torch.is_grad_enabled()
                return inputs

        logits = torch.tensor([[4., 0.], [0., 3.], [2., 0.]])
        targets = torch.tensor([0, 1, 1])
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(logits, targets), batch_size=2)
        model = FixedModel()
        loss, acc = run_epoch(model, loader, torch.nn.CrossEntropyLoss(),
                              torch.device("cpu"), quiet=True)
        self.assertTrue(model.assert_eval)
        self.assertFalse(model.training)
        self.assertAlmostEqual(loss, torch.nn.functional.cross_entropy(logits, targets).item(), places=6)
        self.assertAlmostEqual(acc, 2 / 3)

    def test_frozen_config_applies_scheduler_lr_and_allows_explicit_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formal_config.json"
            path.write_text(json.dumps({"epochs": 150, "batch_size": 128, "split_seed": 1,
                                        "schedulers": {"cosine": {"lr": 0.3}}}), encoding="utf-8")
            with patch("sys.argv", ["train.py", "--config", str(path), "--scheduler", "cosine", "--seed", "3"]):
                config = parse_args()
            self.assertEqual((config.epochs, config.initial_lr, config.seed, config.split_seed), (150, 0.3, 3, 1))
            with patch("sys.argv", ["train.py", "--config", str(path), "--scheduler", "cosine", "--lr", "0.1"]):
                self.assertEqual(parse_args().initial_lr, 0.1)

    def test_momentum_and_weight_decay_remain_fixed_for_entire_schedule(self):
        for name in ("constant", "step", "multistep", "cosine", "warmrestart", "cyclic", "onecycle", "warmup_cosine"):
            with self.subTest(scheduler=name):
                optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1, momentum=0.9, weight_decay=5e-4)
                scheduler, interval, _ = build_scheduler(name, optimizer, epochs=150, steps_per_epoch=3)
                for _ in range(150 * 3 if interval == "batch" else 150):
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    self.assertEqual(optimizer.param_groups[0]["momentum"], 0.9)
                    self.assertEqual(optimizer.param_groups[0]["weight_decay"], 5e-4)

    @patch("train.datasets.CIFAR10", FakeCifar)
    def test_training_seed_does_not_change_split(self):
        first, validation, test = build_loaders(ExperimentConfig(seed=1, split_seed=1))
        second, second_validation, _ = build_loaders(ExperimentConfig(seed=3, split_seed=1))
        self.assertEqual(first.dataset.indices, second.dataset.indices)
        self.assertEqual(validation.dataset.indices, second_validation.dataset.indices)
        self.assertEqual(len(first.dataset), 45000)
        self.assertEqual(len(validation.dataset), 5000)
        self.assertEqual(len(test.dataset), 10000)
        self.assertFalse(set(first.dataset.indices) & set(validation.dataset.indices))
        _, changed, _ = build_loaders(ExperimentConfig(seed=1, split_seed=2))
        self.assertNotEqual(validation.dataset.indices, changed.dataset.indices)


if __name__ == "__main__":
    unittest.main()
