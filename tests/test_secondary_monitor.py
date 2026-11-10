"""Intentional pauses and changes after a previous monitor inspection."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'scripts/monitor_secondary_cifar100.py'
spec = importlib.util.spec_from_file_location('secondary_monitor', PATH)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class MonitorChecks(unittest.TestCase):
    def prepare(self, root):
        (root / 'protocol').mkdir()
        monitor.write_json(root / 'protocol/split.json', {})
        monitor.write_json(root / 'protocol/normalization.json', {})
        monitor.write_json(root / 'manifest.json', {'source_hashes': {},
            'split_sha256': monitor.sha(root/'protocol/split.json'),
            'normalization_sha256': monitor.sha(root/'protocol/normalization.json')})
        monitor.write_json(root/'status.json', {'phase':'paused'})
        monitor.write_json(root/'process.json', {'queue_pid':100,'stdout':str(root/'stdout.log'),'stderr':str(root/'stderr.log')})

    def test_intentional_pause_does_not_require_live_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.prepare(root)
            report=monitor.inspect(root,{})
            self.assertTrue(report['healthy'], report['issues'])
            self.assertEqual(report['phase'],'paused')

    def test_checkpoint_change_is_checked_on_every_inspection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.prepare(root)
            dest=root/'runs/constant/seed16'
            dest.mkdir(parents=True)
            for name in ['best.pt','last.pt']:(dest/name).write_bytes(b'original')
            monitor.write_json(dest/'training_complete.json', {'training_time_seconds':100,
                'test_acc':None,'best_sha256':monitor.sha(dest/'best.pt'),'last_sha256':monitor.sha(dest/'last.pt')})
            monitor.write_json(dest/'audit.json', {'passed':True})
            for i in range(6):(dest/f'{i}.png').touch()
            def saved_rows(path):
                return ([{'epoch':str(i)} for i in range(1,151)] if path.name=='epochs.csv'
                        else [{'update':str(i)} for i in range(1,52801)])
            with patch.object(monitor,'rows',side_effect=saved_rows):
                self.assertTrue(monitor.inspect(root,{})['healthy'])
                (dest/'best.pt').write_bytes(b'changed!')
                report=monitor.inspect(root,{})
                self.assertFalse(report['healthy'])
                self.assertTrue(any('Checkpoint changed' in x for x in report['issues']))


if __name__=='__main__':
    unittest.main()
