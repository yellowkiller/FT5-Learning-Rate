"""Ensure optimizer, schedule and random-stream continuation match an uninterrupted run."""
import copy
import importlib.util
from pathlib import Path
import random
import unittest
import numpy as np
import torch

path = Path(__file__).resolve().parents[1] / 'scripts/run_secondary_cifar100_recovery.py'
spec = importlib.util.spec_from_file_location('recovery', path)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class RecoveryChecks(unittest.TestCase):
    def test_random_optimizer_schedule_continuation(self):
        torch.set_num_threads(1)
        torch.manual_seed(2468); random.seed(2468); np.random.seed(2468)

        def build():
            model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
            optimizer = torch.optim.SGD(model.parameters(), lr=.03, momentum=.9)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.03, total_steps=12, cycle_momentum=False)
            return model, optimizer, scheduler

        def steps(model, optimizer, scheduler, count):
            trace = []
            for _ in range(count):
                x = torch.randn(5, 3) * (random.random()+float(np.random.random()))
                y = torch.randn(5, 1)
                optimizer.zero_grad(); loss = ((model(x)-y)**2).mean()
                trace.append((optimizer.param_groups[0]['lr'], float(loss.detach())))
                loss.backward(); optimizer.step(); scheduler.step()
            return trace

        model, optimizer, scheduler = build()
        steps(model, optimizer, scheduler, 6)
        state = copy.deepcopy({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()})
        expected = steps(model, optimizer, scheduler, 6)
        resumed_model, resumed_optimizer, resumed_scheduler = build()
        resumed_model.load_state_dict(state['model'])
        resumed_optimizer.load_state_dict(state['optimizer'])
        resumed_scheduler.load_state_dict(state['scheduler'])
        recovery.restore_rng(state)
        observed = steps(resumed_model, resumed_optimizer, resumed_scheduler, 6)
        self.assertEqual(expected, observed)
        for a, b in zip(model.parameters(), resumed_model.parameters()):
            self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
