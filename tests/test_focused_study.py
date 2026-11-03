import csv
import io
import unittest

import torch

from train import better_checkpoint, run_epoch
from schedulers import build_scheduler


class FocusedStudyTests(unittest.TestCase):
    def test_checkpoint_ties_use_loss_then_earlier_epoch(self):
        self.assertTrue(better_checkpoint(.95, .2, .94, .1))
        self.assertTrue(better_checkpoint(.95, .2, .95, .3))
        self.assertFalse(better_checkpoint(.95, .2, .95, .2))
        self.assertFalse(better_checkpoint(.94, .1, .95, .3))

    def test_lr_log_records_updates_before_scheduler_step(self):
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        sched, interval, _ = build_scheduler("cyclic", opt, epochs=150, steps_per_epoch=2,
                                            options={"base_lr": .001, "step_size_up": 2})
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.ones(4, 2), torch.tensor([0, 1, 0, 1])), batch_size=2)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=["epoch", "batch", "update", "lr", "momentum"])
        writer.writeheader()
        run_epoch(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"), opt, sched,
                  interval, True, lr_writer=writer, epoch=1)
        stream.seek(0)
        rows = list(csv.DictReader(stream))
        self.assertEqual([int(r["update"]) for r in rows], [1, 2])
        self.assertAlmostEqual(float(rows[0]["lr"]), .001)
        self.assertAlmostEqual(float(rows[1]["lr"]), .0505)
        self.assertTrue(all(float(r["momentum"]) == .9 for r in rows))

    def test_nonfinite_loss_fails_instead_of_producing_results(self):
        model = torch.nn.Linear(2, 2)
        loader = [(torch.full((1, 2), float("nan")), torch.tensor([0]))]
        with self.assertRaises(FloatingPointError):
            run_epoch(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"), quiet=True)


if __name__ == "__main__":
    unittest.main()
