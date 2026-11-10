"""Campaign-level checks: incomplete-test gate and missing-threshold statistics."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'scripts/run_secondary_cifar100.py'
spec = importlib.util.spec_from_file_location('secondary', PATH)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class SecondaryChecks(unittest.TestCase):
    def test_incomplete_campaign_cannot_access_test_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            s.write_json(root / 'selected_checkpoints_locked.json', {'models': []})
            with patch.object(s, 'source_check', return_value={'matrix': []}), patch.object(s.datasets, 'CIFAR100') as dataset:
                with self.assertRaises(ValueError):
                    s.evaluate_locked(root)
                dataset.assert_not_called()

    def test_duplicate_checkpoint_lock_cannot_access_test_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            matrix = [{'scheduler': policy, 'seed': seed} for policy in s.POLICIES for seed in [16,17,18]]
            s.write_json(root / 'selected_checkpoints_locked.json', {'models': [{'policy':'constant','seed':16}]*24})
            with patch.object(s, 'source_check', return_value={'matrix':matrix}), patch.object(s.datasets, 'CIFAR100') as dataset:
                with self.assertRaises(ValueError):
                    s.evaluate_locked(root)
                dataset.assert_not_called()

    def test_refused_existing_root_preserves_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            s.write_json(root / 'status.json', {'phase':'complete','completed':24})
            original = (root / 'status.json').read_bytes()
            with self.assertRaises(FileExistsError):
                s.queue(root, root / 'unused-data', False)
            self.assertEqual((root / 'status.json').read_bytes(), original)

    def test_summary_reports_nonattainment_and_known_auc(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for seed in [16, 17, 18]:
                for policy in s.POLICIES:
                    dest = root / 'runs' / policy / f'seed{seed}'
                    s.write_json(dest / 'training_complete.json', {
                        'best_val_acc': .6, 'training_time_seconds': 3600})
                    s.write_json(dest / 'test_result.json', {'test_acc': .55})
                    s.write_csv(dest / 'epochs.csv', [{'val_acc': .6} for _ in range(150)])
            s.summarize(root)
            rows = s.read_csv(root / 'summary/summary.csv')
            self.assertEqual(len(rows), 8)
            for row in rows:
                self.assertEqual(float(row['auc_pct_mean']), 60.)
                self.assertEqual(row['first50_epoch_n'], '3')
                self.assertEqual(float(row['first50_epoch_mean']), 1.)
                self.assertEqual(float(row['sustained50_confirmed_epoch_mean']), 5.)
                self.assertEqual(row['first70_epoch_n'], '0')
                self.assertEqual(row['first70_epoch_mean'], '')
                self.assertEqual(float(row['test_pct_sample_sd']), 0.)
            self.assertEqual(len(s.read_csv(root / 'summary/paired_differences.csv')), 15)


if __name__ == '__main__':
    unittest.main()
