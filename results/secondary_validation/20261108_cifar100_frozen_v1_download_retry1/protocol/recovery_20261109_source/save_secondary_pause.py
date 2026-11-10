"""Audit and record a user-requested pause without training or test evaluation."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import sys
import torch

root = Path(sys.argv[1]).resolve()
frozen = root / 'protocol/source_snapshot/scripts/run_secondary_cifar100.py'
api = runpy.run_path(str(frozen))
manifest = api['source_check'](root)
complete, pending, partial = [], [], []

for spec in manifest['matrix']:
    policy, seed = spec['scheduler'], int(spec['seed'])
    dest = root / 'runs' / policy / f'seed{seed}'
    if (dest / 'training_complete.json').exists():
        c = api['audit'](root, policy, seed)
        complete.append(c)
    elif dest.exists():
        records = api['read_csv'](dest / 'epochs.csv')
        item = {'policy': policy, 'seed': seed, 'directory': str(dest),
                'epoch_rows': len(records), 'last_logged_epoch': int(records[-1]['epoch']) if records else 0,
                'checkpoints': {}}
        for filename in ['best.pt', 'last.pt']:
            path = dest / filename
            if path.exists():
                state = torch.load(path, map_location='cpu', weights_only=False)
                item['checkpoints'][filename] = {'epoch': state['epoch'], 'sha256': api['sha'](path),
                    'optimizer_state_saved': 'optimizer_state' in state,
                    'rng_state_saved': all(k in state for k in ['torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng'])}
                del state
        item['resume_constraint'] = ('Preserved incomplete run. Existing --resume refuses this directory. '
            'Any future checkpoint continuation needs an explicit versioned recovery implementation; '
            'do not overwrite or silently restart. Epoch CSV can extend past the latest full-state checkpoint.')
        partial.append(item)
        pending.append(spec)
    else:
        pending.append(spec)

assert len(complete) == 12
assert len(list((root / 'runs').rglob('test_result.json'))) == 0
api['write_csv'](root / 'summary/completed_training_at_pause.csv', complete)
saved = {'paused_at': datetime.now(timezone.utc).isoformat(), 'reason': 'User requested stop after current run',
         'completed': len(complete), 'total': 24, 'remaining_configurations': pending,
         'partial_runs_preserved': partial, 'test_evaluations': 0,
         'target_run_completed': 'cyclic/seed17', 'training_and_queue_stopped': True,
         'frozen_manifest_sha256': api['sha'](root / 'manifest.json'),
         'protocol_and_training_source_unchanged': True}
api['write_json'](root / 'pause_state.json', saved)
api['write_json'](root / 'status.json', {'phase': 'paused', 'completed': len(complete), 'total': 24,
    'updated_at': saved['paused_at'], 'reason': saved['reason'], 'partial_runs': partial,
    'target_run_completed': 'cyclic/seed17', 'next_policy': pending[0]['scheduler'],
    'next_seed': int(pending[0]['seed']), 'test_evaluations': 0, 'pause_state': str(root / 'pause_state.json')})
api['write_json'](root / 'monitor/latest.json', {'checked_at': saved['paused_at'], 'phase': 'paused',
    'completed': len(complete), 'total': 24, 'monitor_enabled': False, 'issues': [],
    'reason': 'User-requested pause; normal hourly monitor stopped with training'})
print(json.dumps(saved, indent=2))
