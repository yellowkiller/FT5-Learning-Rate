"""Versioned checkpoint continuation; leaves original frozen training source intact."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn


def load_api(root):
    return runpy.run_path(str(root / 'protocol/source_snapshot/scripts/run_secondary_cifar100.py'))


def restore_rng(state):
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])


def continuation(root, policy, seed):
    api = load_api(root)
    api['source_check'](root)
    assert (policy, seed) == ('onecycle', 17)
    dest = root / 'runs' / policy / f'seed{seed}'
    backup = root / 'recovery_20261109' / 'onecycle_seed17_before_resume'
    assert not (dest / 'training_complete.json').exists()
    assert not backup.exists(), 'Do not automatically repeat a checkpoint recovery'
    pause = api['read_json'](root / 'pause_state.json')['partial_runs_preserved'][0]
    for filename, info in pause['checkpoints'].items():
        assert api['sha'](dest / filename) == info['sha256']
    state = torch.load(dest / 'last.pt', map_location='cpu', weights_only=False)
    best = torch.load(dest / 'best.pt', map_location='cpu', weights_only=False)
    assert state['epoch'] == 30 and state['best_epoch'] == best['epoch'] == 29
    epochs = api['read_csv'](dest / 'epochs.csv')
    prefix = [r for r in epochs if int(r['epoch']) <= state['epoch']]
    assert [int(r['epoch']) for r in prefix] == list(range(1, 31))
    lrprefix = [r for r in api['read_csv'](dest / 'lr_updates.csv') if int(r['update']) <= 10560]
    assert len(lrprefix) == 10560
    assert state['scheduler_state']['last_epoch'] == 10560
    assert abs(state['optimizer_state']['param_groups'][0]['lr'] - api['analytic_trace'](policy)[10560]) < 1e-10
    # Preserve every pre-pause artifact before selecting the checkpoint-consistent prefix.
    shutil.copytree(dest, backup)
    backup_hashes = {p.name: api['sha'](p) for p in backup.iterdir() if p.is_file()}
    record = {'protocol': 'checkpoint-continuation-v1', 'started_at': api['now'](),
              'policy': policy, 'seed': seed, 'checkpoint_epoch': 30, 'first_resumed_epoch': 31,
              'checkpoint_sha256': api['sha'](backup / 'last.pt'),
              'backup': str(backup), 'backup_hashes': backup_hashes,
              'recovery_script': str(Path(__file__).resolve()), 'recovery_script_sha256': api['sha'](__file__),
              'original_source_unchanged': True, 'test_evaluation': False,
              'timing': 'sum of saved epochs1-30 plus resumed active time; pause excluded',
              'comparison_limit': 'resumed execution is segmented; do not claim uninterrupted wall-clock equivalence'}
    api['write_json'](dest / 'recovery.json', record)
    api['write_csv'](dest / 'epochs.csv', prefix)
    api['write_csv'](dest / 'lr_updates.csv', lrprefix)
    torch.set_num_threads(4)
    api['set_seed'](seed)
    train = api['data_loader'](root, seed, True)
    validation = api['data_loader'](root, seed, False)
    model = api['build_model']('resnet18', num_classes=100).cuda()
    model.load_state_dict(state['model_state'])
    optimizer = torch.optim.SGD(model.parameters(), lr=.03, momentum=.9, weight_decay=.0005)
    scheduler, step, _ = api['build_scheduler'](policy, optimizer, epochs=150, steps_per_epoch=352, options={'pct_start': .3})
    optimizer.load_state_dict(state['optimizer_state'])
    scheduler.load_state_dict(state['scheduler_state'])
    assert scheduler.last_epoch == 10560
    assert abs(optimizer.param_groups[0]['lr'] - api['analytic_trace'](policy)[10560]) < 1e-10
    best_acc, best_loss, best_epoch = best['val_acc'], best['val_loss'], best['epoch']
    metadata = dict(state['metadata'])
    metadata['checkpoint_recovery'] = record
    api['write_json'](dest / 'metadata.json', metadata)
    criterion = nn.CrossEntropyLoss()
    # Construction consumes randomness. Restore only after every constructor/load.
    restore_rng(state)
    del state, best
    api['sync'](); start = time.perf_counter()
    with (dest / 'epochs.csv').open('a', newline='', encoding='utf8') as ef, (dest / 'lr_updates.csv').open('a', newline='', encoding='utf8') as lf:
        ew = csv.DictWriter(ef, fieldnames=list(prefix[0]))
        lw = csv.DictWriter(lf, fieldnames=list(lrprefix[0]))
        for epoch in range(31, 151):
            api['sync'](); epoch_start = time.perf_counter()
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_acc = api['run_epoch'](model, train, criterion, 'cuda', optimizer, scheduler, step, True, lw, epoch)
            val_loss, val_acc = api['run_epoch'](model, validation, criterion, 'cuda', quiet=True)
            assert all(math.isfinite(x) for x in [train_loss, train_acc, val_loss, val_acc])
            if api['better_checkpoint'](val_acc, val_loss, best_acc, best_loss):
                best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
                torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc,
                            'val_loss': val_loss, 'metadata': metadata}, dest / 'best.pt')
            if epoch % 10 == 0:
                tmp = dest / 'last.tmp'
                torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': scheduler.state_dict(), 'epoch': epoch, 'best_epoch': best_epoch,
                            'metadata': metadata, 'torch_rng': torch.get_rng_state(),
                            'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(),
                            'numpy_rng': np.random.get_state()}, tmp)
                tmp.replace(dest / 'last.pt')
            api['sync'](); elapsed = time.perf_counter()-epoch_start
            ew.writerow(dict(zip(list(prefix[0]), [epoch, lr, train_loss, train_acc, val_loss, val_acc, elapsed])))
            ef.flush(); lf.flush()
            api['write_json'](root / 'progress.json', {'policy': policy, 'seed': seed, 'epoch': epoch,
                              'epoch_seconds': elapsed, 'updated_at': api['now'](), 'resumed_from_epoch': 30})
            print(f'{policy} seed{seed} resumed epoch {epoch}/150 val={val_acc:.4f} seconds={elapsed:.2f}', flush=True)
    api['sync'](); active = time.perf_counter()-start
    prior = sum(float(r['epoch_seconds']) for r in prefix)
    api['write_json'](dest / 'training_complete.json', {'policy': policy, 'seed': seed, 'epochs': 150,
        'best_epoch': best_epoch, 'best_val_acc': best_acc, 'best_val_loss': best_loss, 'final_val_acc': val_acc,
        'training_time_seconds': prior+active, 'finished_at': api['now'](),
        'best_sha256': api['sha'](dest/'best.pt'), 'last_sha256': api['sha'](dest/'last.pt'), 'test_acc': None,
        'resumed': True, 'resume_checkpoint_epoch': 30, 'pre_pause_active_seconds': prior,
        'resumed_active_seconds': active, 'wall_clock_pause_excluded': True})


def queue(root):
    api = load_api(root)
    try:
        api['source_check'](root)
        for spec in api['read_json'](root / 'manifest.json')['matrix'][:12]:
            api['audit'](root, spec['scheduler'], int(spec['seed']))
        dest = root / 'runs/onecycle/seed17'
        if not (dest/'training_complete.json').exists():
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(root), '--run', 'onecycle', '--seed', '17']
            log = root / 'logs/13_onecycle_seed17_resumed_20261109.log'
            with log.open('x', encoding='utf8') as output:
                child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, cwd=root/'protocol/source_snapshot')
                api['write_json'](root/'status.json', {'phase': 'training', 'completed': 12, 'total': 24,
                    'current_policy': 'onecycle', 'seed': 17, 'training_pid': child.pid,
                    'command': command, 'log': str(log), 'updated_at': api['now'](), 'resumed_from_epoch': 30})
                code = child.wait()
            if code: raise RuntimeError(f'Checkpoint continuation failed ({code}); inspect {log}; no automatic retry')
        api['audit'](root, 'onecycle', 17)
        api['plots'](root, 'onecycle', 17)
        # Frozen queue audits/skips complete runs and starts only the remaining matrix entries.
        api['queue'](root, Path(api['read_json'](root/'manifest.json')['data_dir']), True)
    except BaseException as exc:
        previous = api['read_json'](root/'status.json')
        api['write_json'](root/'status.json', {**previous, 'phase': 'error', 'error': repr(exc), 'updated_at': api['now']()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', choices=['onecycle'])
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.run:
        continuation(root, args.run, args.seed)
    else:
        import msvcrt
        lock = root.parent / (root.name+'.queue.lock')
        with lock.open('r+b') as handle:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            queue(root)
