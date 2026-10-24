import unittest

import torch

from schedulers import build_scheduler


class DiagnosticOptionsTests(unittest.TestCase):
    def test_restart_cycle_completes_at_budget_boundary(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        scheduler, _, metadata = build_scheduler("warmrestart", optimizer, epochs=150,
                                                 steps_per_epoch=352, options={"T_0": 50})
        values = []
        for _ in range(150):
            values.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(values[50], 0.1)
        self.assertLess(values[-1], 0.0001)
        self.assertIn("T_0=50", metadata["scheduler_params"])

    def test_cyclic_lower_bound_and_fixed_momentum(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1, momentum=0.9)
        scheduler, _, _ = build_scheduler("cyclic", optimizer, epochs=30, steps_per_epoch=3,
                                          options={"base_lr": 0.001})
        for _ in range(90):
            self.assertGreaterEqual(optimizer.param_groups[0]["lr"], 0.001)
            self.assertEqual(optimizer.param_groups[0]["momentum"], 0.9)
            optimizer.step()
            scheduler.step()

    def test_onecycle_peak_and_reject_unknown_parameters(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        scheduler, _, _ = build_scheduler("onecycle", optimizer, epochs=30, steps_per_epoch=10,
                                          options={"pct_start": 0.1})
        values = []
        for _ in range(300):
            values.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertEqual(values.index(max(values)), 29)
        self.assertAlmostEqual(values[-1], 4e-7)
        with self.assertRaises(ValueError):
            build_scheduler("cosine", optimizer, epochs=30, steps_per_epoch=10,
                            options={"misspelled": 1})


if __name__ == "__main__":
    unittest.main()
