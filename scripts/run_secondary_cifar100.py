"""Fixed CIFAR-100 transfer campaign. No tuning or test access during training."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
if sys.flags.optimize:
    raise RuntimeError('This protocol requires Python without -O: integrity assertions must remain enabled')
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models.resnet import build_model
from schedulers import build_scheduler
from train import better_checkpoint, run_epoch
from utils.seed import set_seed

POLICIES = {
    'constant': {'lr': .1, 'options': {}},
    'step': {'lr': .1, 'options': {}},
    'multistep': {'lr': .1, 'options': {}},
    'cosine': {'lr': .1, 'options': {}},
    'warmrestart': {'lr': .1, 'options': {'T_0': 50, 'T_mult': 2}},
    'warmup_cosine': {'lr': .1, 'options': {}},
    'cyclic': {'lr': .1, 'options': {'base_lr': .01, 'step_size_up': 704}},
    'onecycle': {'lr': .03, 'options': {'pct_start': .3}},
}
SOURCES = ['scripts/run_secondary_cifar100.py', 'train.py', 'config.py', 'schedulers.py',
           'models/resnet.py', 'utils/seed.py', 'utils/metrics.py', 'utils/logger.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sync():
    torch.cuda.synchronize()


def schedule_trace(policy):
    p = torch.nn.Parameter(torch.zeros(1))
    cfg = POLICIES[policy]
    opt = torch.optim.SGD([p], lr=cfg['lr'], momentum=.9, weight_decay=.0005)
    sched, step, meta = build_scheduler(policy, opt, epochs=150, steps_per_epoch=352, options=cfg['options'])
    values = []
    for epoch in range(150):
        for batch in range(352):
            values.append(opt.param_groups[0]['lr'])
            opt.step()
            if sched is not None and step == 'batch':
                sched.step()
        if sched is not None and step == 'epoch':
            sched.step()
    return np.asarray(values), meta


def analytic_trace(policy):
    # Independent formula check at the LR used before optimizer.step().
    u = np.arange(52800, dtype=float)
    e = np.floor(u / 352)
    if policy == 'constant': return np.full(52800, .1)
    if policy == 'step': return .1 * .1 ** np.floor(e / 50)
    if policy == 'multistep': return .1 * .1 ** ((e >= 75).astype(int) + (e >= 112).astype(int))
    if policy == 'cosine': return .05 * (1 + np.cos(np.pi * e / 150))
    if policy == 'warmrestart':
        t = np.where(e < 50, e, e - 50)
        period = np.where(e < 50, 50, 100)
        return .05 * (1 + np.cos(np.pi * t / period))
    if policy == 'warmup_cosine':
        return np.where(e < 7, .1 * (.01 + .99 * e / 7), .05 * (1 + np.cos(np.pi * (e - 7) / 143)))
    if policy == 'cyclic':
        cycle = np.floor(1 + u / 1408)
        x = np.abs(u / 704 - 2 * cycle + 1)
        return .01 + .09 * np.maximum(0, 1 - x) / 2 ** (cycle - 1)
    peak = .3 * 52800 - 1
    start, minimum = .03 / 25, .03 / 25 / 10000
    return np.where(u <= peak,
                    .03 + (start - .03) * (np.cos(np.pi * u / peak) + 1) / 2,
                    minimum + (.03 - minimum) * (np.cos(np.pi * (u - peak) / (52799 - peak)) + 1) / 2)


def preflight():
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    errors = {}
    for policy in POLICIES:
        actual, _ = schedule_trace(policy)
        error = float(np.max(np.abs(actual - analytic_trace(policy))))
        assert len(actual) == 52800 and np.all(np.isfinite(actual)) and error < 1e-10, (policy, error)
        errors[policy] = error
    assert better_checkpoint(.6, 1, .5, .5)
    assert better_checkpoint(.6, .9, .6, 1)
    assert not better_checkpoint(.6, 1, .6, 1)
    assert not better_checkpoint(.59, .1, .6, 1)
    set_seed(16000)  # Implementation-only seed, never a result.
    model = build_model('resnet18', num_classes=100).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    x = torch.randn(4, 3, 32, 32, device='cuda')
    logits = model(x)
    assert logits.shape == (4, 100)
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 25, 50, 99], device='cuda'))
    loss.backward(); optimizer.step(); sync()
    return {'checked_at': now(), 'analytic_schedule_max_errors': errors,
            'logits_shape': [4, 100], 'finite_backward_loss': float(loss.detach()),
            'checkpoint_tie_checks': 'pass', 'test_data_accessed': False}


def data_loader(root, seed, training):
    manifest = read_json(root / 'manifest.json')
    norm = read_json(root / 'protocol/normalization.json')
    split = read_json(root / 'protocol/split.json')
    augment = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if training else []
    transform = transforms.Compose(augment + [transforms.ToTensor(), transforms.Normalize(norm['mean'], norm['std'])])
    ds = datasets.CIFAR100(manifest['data_dir'], train=True, download=False, transform=transform)
    ds = Subset(ds, split['train_indices' if training else 'validation_indices'])
    return DataLoader(ds, batch_size=128, shuffle=training, num_workers=0, pin_memory=True)


def source_check(root):
    manifest = read_json(root / 'manifest.json')
    snap = root / 'protocol/source_snapshot'
    for name, digest in manifest['source_hashes'].items():
        assert sha(snap / name) == digest, f'Source changed: {name}'
    assert sha(root / 'protocol/split.json') == manifest['split_sha256']
    assert sha(root / 'protocol/normalization.json') == manifest['normalization_sha256']
    return manifest


def train_run(root, policy, seed):
    manifest = source_check(root)
    assert seed in [16, 17, 18] and policy in POLICIES
    dest = root / 'runs' / policy / f'seed{seed}'
    dest.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); set_seed(seed)
    train = data_loader(root, seed, True)
    validation = data_loader(root, seed, False)
    assert len(train.dataset) == 45000 and len(validation.dataset) == 5000 and len(train) == 352
    model = build_model('resnet18', num_classes=100).cuda()
    criterion = nn.CrossEntropyLoss()
    cfg = POLICIES[policy]
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=.9, weight_decay=.0005)
    sched, step, meta = build_scheduler(policy, optimizer, epochs=150, steps_per_epoch=352, options=cfg['options'])
    metadata = {'policy': policy, 'seed': seed, 'dataset': 'cifar100', 'classes': 100,
                'epochs': 150, 'scheduler': meta, 'split_sha256': manifest['split_sha256'],
                'normalization_sha256': manifest['normalization_sha256'], 'source_hashes': manifest['source_hashes'],
                'test_evaluation': False, 'started_at': now(), 'gpu': torch.cuda.get_device_name(0)}
    write_json(dest / 'metadata.json', metadata)
    fields = ['epoch', 'lr', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'epoch_seconds']
    best_acc, best_loss, best_epoch = -1., float('inf'), 0
    sync(); start = time.perf_counter()
    with (dest / 'epochs.csv').open('w', newline='', encoding='utf8') as ef, (dest / 'lr_updates.csv').open('w', newline='', encoding='utf8') as lf:
        ew = csv.DictWriter(ef, fieldnames=fields); ew.writeheader()
        lw = csv.DictWriter(lf, fieldnames=['epoch', 'batch', 'update', 'lr', 'momentum']); lw.writeheader()
        for epoch in range(1, 151):
            sync(); epoch_start = time.perf_counter()
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_acc = run_epoch(model, train, criterion, 'cuda', optimizer, sched, step, True, lw, epoch)
            val_loss, val_acc = run_epoch(model, validation, criterion, 'cuda', quiet=True)
            assert all(math.isfinite(x) for x in [train_loss, train_acc, val_loss, val_acc])
            if sched is not None and step == 'epoch': sched.step()
            improved = better_checkpoint(val_acc, val_loss, best_acc, best_loss)
            if improved:
                best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
                torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc,
                            'val_loss': val_loss, 'metadata': metadata}, dest / 'best.pt')
            if epoch % 10 == 0:
                tmp = dest / 'last.tmp'
                torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': sched.state_dict() if sched else None, 'epoch': epoch,
                            'best_epoch': best_epoch, 'metadata': metadata, 'torch_rng': torch.get_rng_state(),
                            'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(),
                            'numpy_rng': np.random.get_state()}, tmp)
                tmp.replace(dest / 'last.pt')
            sync(); elapsed = time.perf_counter() - epoch_start
            ew.writerow(dict(zip(fields, [epoch, lr, train_loss, train_acc, val_loss, val_acc, elapsed])))
            ef.flush(); lf.flush()
            write_json(root / 'progress.json', {'policy': policy, 'seed': seed, 'epoch': epoch,
                       'epoch_seconds': elapsed, 'updated_at': now()})
            print(f'{policy} seed{seed} epoch {epoch}/150 val={val_acc:.4f} seconds={elapsed:.2f}', flush=True)
    sync(); total = time.perf_counter() - start
    write_json(dest / 'training_complete.json', {'policy': policy, 'seed': seed, 'epochs': 150,
               'best_epoch': best_epoch, 'best_val_acc': best_acc, 'best_val_loss': best_loss,
               'final_val_acc': val_acc, 'training_time_seconds': total, 'finished_at': now(),
               'best_sha256': sha(dest / 'best.pt'), 'last_sha256': sha(dest / 'last.pt'),
               'test_acc': None})


def audit(root, policy, seed):
    manifest = source_check(root)
    dest = root / 'runs' / policy / f'seed{seed}'
    complete = read_json(dest / 'training_complete.json')
    rows = read_csv(dest / 'epochs.csv')
    lrrows = read_csv(dest / 'lr_updates.csv')
    assert [int(r['epoch']) for r in rows] == list(range(1, 151))
    assert [int(r['update']) for r in lrrows] == list(range(1, 52801))
    assert all(int(r['epoch']) == i // 352 + 1 and int(r['batch']) == i % 352 + 1 for i, r in enumerate(lrrows))
    assert all(float(r['momentum']) == .9 for r in lrrows)
    assert np.max(np.abs(np.array([float(r['lr']) for r in lrrows]) - analytic_trace(policy))) < 1e-10
    selected = min(rows, key=lambda r: (-float(r['val_acc']), float(r['val_loss']), int(r['epoch'])))
    assert int(selected['epoch']) == complete['best_epoch']
    assert float(selected['val_acc']) == complete['best_val_acc']
    assert float(selected['val_loss']) == complete['best_val_loss']
    assert complete['test_acc'] is None and sha(dest / 'best.pt') == complete['best_sha256']
    assert sha(dest / 'last.pt') == complete['last_sha256']
    metadata = read_json(dest / 'metadata.json')
    assert metadata['split_sha256'] == manifest['split_sha256'] and metadata['classes'] == 100
    best = torch.load(dest / 'best.pt', map_location='cpu', weights_only=False)
    last = torch.load(dest / 'last.pt', map_location='cpu', weights_only=False)
    assert best['epoch'] == complete['best_epoch'] and last['epoch'] == 150
    assert best['model_state']['linear.weight'].shape == (100, 512)
    write_json(dest / 'audit.json', {'passed': True, 'checked_at': now(), 'epoch_rows': 150, 'lr_rows': 52800})
    return complete


def plots(root, policy, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest = root / 'runs' / policy / f'seed{seed}'
    rows = read_csv(dest / 'epochs.csv'); x = np.arange(1, 151)
    for name, columns, ylabel, start in [
        ('loss', ['train_loss', 'val_loss'], 'Cross entropy', 0),
        ('accuracy', ['train_acc', 'val_acc'], 'Accuracy', 0),
        ('validation_loss', ['val_loss'], 'Validation cross entropy', 0),
        ('validation_accuracy', ['val_acc'], 'Validation accuracy', 0),
        ('late_validation', ['val_acc'], 'Validation accuracy', 140)]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for c in columns: ax.plot(x[start:], [float(r[c]) for r in rows[start:]], label=c)
        ax.set(xlabel='Epoch', ylabel=ylabel, title=f'{policy} seed {seed}'); ax.legend(); fig.tight_layout()
        fig.savefig(dest / f'{name}.png', dpi=150); plt.close(fig)
    lr = read_csv(dest / 'lr_updates.csv')
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot([int(r['update']) for r in lr], [float(r['lr']) for r in lr])
    ax.set(xlabel='Optimizer update', ylabel='LR', title=f'{policy} seed {seed}'); fig.tight_layout()
    fig.savefig(dest / 'learning_rate.png', dpi=150); plt.close(fig)


def evaluate_locked(root):
    manifest = source_check(root)
    lock = read_json(root / 'selected_checkpoints_locked.json')
    expected = {(r['scheduler'], int(r['seed'])) for r in manifest['matrix']}
    observed = [(r['policy'], int(r['seed'])) for r in lock['models']]
    if len(observed) != 24 or len(expected) != 24 or set(observed) != expected:
        raise ValueError('Checkpoint lock must cover all 24 unique frozen runs')
    # Require all training audits and unchanged best hashes before any test read.
    for item in lock['models']:
        if item['directory'] != f"runs/{item['policy']}/seed{item['seed']}":
            raise ValueError('Checkpoint directory does not match its frozen run')
        dest = root / item['directory']
        complete = read_json(dest / 'training_complete.json')
        if (complete['policy'], complete['seed'], complete['epochs']) != (item['policy'], int(item['seed']), 150):
            raise ValueError('Training completion does not match the locked run')
        if complete['test_acc'] is not None or complete['best_sha256'] != item['sha256']:
            raise ValueError('Training record includes test selection or a changed checkpoint')
        assert read_json(dest / 'audit.json')['passed'] and sha(dest / 'best.pt') == item['sha256']
    manifest = read_json(root / 'manifest.json'); norm = read_json(root / 'protocol/normalization.json')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm['mean'], norm['std'])])
    test = datasets.CIFAR100(manifest['data_dir'], train=False, download=False, transform=transform)
    assert len(test) == 10000 and len(test.classes) == 100
    loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    torch.set_num_threads(4)
    for item in lock['models']:
        dest = root / item['directory']
        if (dest / 'test_result.json').exists():
            assert read_json(dest / 'test_result.json')['best_sha256'] == item['sha256']
            continue
        # A crashed test requires investigation, never automatic re-evaluation.
        with (dest / 'test_started.json').open('x', encoding='utf8') as f: json.dump({'started_at': now()}, f)
        model = build_model('resnet18', num_classes=100).cuda()
        checkpoint = torch.load(dest / 'best.pt', map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        loss, acc = run_epoch(model, loader, nn.CrossEntropyLoss(), 'cuda', quiet=True)
        write_json(dest / 'test_result.json', {'test_loss': loss, 'test_acc': acc,
                   'best_sha256': item['sha256'], 'evaluated_at': now(), 'selection': 'locked validation best'})
        del model, checkpoint


def summarize(root):
    per_seed = []
    for seed in [16, 17, 18]:
        for policy in POLICIES:
            dest = root / 'runs' / policy / f'seed{seed}'
            c = read_json(dest / 'training_complete.json'); t = read_json(dest / 'test_result.json')
            vals = [float(r['val_acc']) * 100 for r in read_csv(dest / 'epochs.csv')]
            row = {'policy': policy, 'seed': seed, 'best_validation_pct': c['best_val_acc'] * 100,
                   'final_validation_pct': vals[-1], 'test_pct': t['test_acc'] * 100,
                   'late_validation_sd_pp': statistics.stdev(vals[-10:]),
                   'auc_pct': sum((a + b) / 2 for a, b in zip(vals[:-1], vals[1:])) / 149,
                   'auc30_pct': sum((a + b) / 2 for a, b in zip(vals[:29], vals[1:30])) / 29,
                   'training_minutes': c['training_time_seconds'] / 60}
            for threshold in [50, 60, 70]:
                for kind in ['first', 'sustained', 'terminal']:
                    hits = [i + 1 for i in range(150) if
                            (vals[i] >= threshold if kind == 'first' else
                             (i <= 145 and min(vals[i:i+5]) >= threshold) if kind == 'sustained' else
                             min(vals[i:]) >= threshold)]
                    epoch = hits[0] if hits else None
                    row[f'{kind}{threshold}_epoch'] = epoch
                    row[f'{kind}{threshold}_updates'] = epoch * 352 if epoch else None
                row[f'sustained{threshold}_confirmed_epoch'] = row[f'sustained{threshold}_epoch'] + 4 if row[f'sustained{threshold}_epoch'] else None
            per_seed.append(row)
    write_csv(root / 'summary/per_seed.csv', per_seed)
    summary = []
    for policy in POLICIES:
        group = [r for r in per_seed if r['policy'] == policy]; record = {'policy': policy}
        for k in per_seed[0]:
            if k in ['policy', 'seed']: continue
            values = [r[k] for r in group if r[k] is not None]
            record[k + '_n'] = len(values)
            record[k + '_mean'] = statistics.mean(values) if values else None
            record[k + '_sample_sd'] = statistics.stdev(values) if len(values) > 1 else None
        summary.append(record)
    write_csv(root / 'summary/summary.csv', summary)
    paired = []
    for left, right in [('cyclic', 'warmrestart'), ('cyclic', 'onecycle'),
                        ('warmup_cosine', 'cosine'), ('onecycle', 'warmrestart'), ('step', 'multistep')]:
        for seed in [16, 17, 18]:
            a = next(r for r in per_seed if r['policy'] == left and r['seed'] == seed)
            b = next(r for r in per_seed if r['policy'] == right and r['seed'] == seed)
            row = {'left': left, 'right': right, 'seed': seed}
            for key in a:
                if key in ['policy', 'seed']: continue
                row[key + '_delta'] = a[key] - b[key] if a[key] is not None and b[key] is not None else None
            paired.append(row)
    write_csv(root / 'summary/paired_differences.csv', paired)
    (root / 'README.md').write_text('CIFAR-100 frozen transfer study\n\n24 fixed runs. Validation-selected checkpoints were locked before test evaluation. '
        'All accuracies and AUC in summary tables are percentages. SD is sample SD, n=3 unless attainment counts state otherwise. '
        'Do not infer population superiority, equal historical tuning budgets or a speedup over shared-GPU primary times. '
        'See protocol/design.md, manifest.json and selected_checkpoints_locked.json for provenance.\n', encoding='utf8')


def prepare(root, data_dir):
    if root.exists(): raise FileExistsError(f'Refusing existing root: {root}')
    # Audit historical run registries by exact seed column, not substring matching.
    conflicts = []
    for path in (BASE / 'results').rglob('*.csv'):
        if 'source_snapshot' in path.parts or path.name not in ['results.csv', 'all_runs.csv', 'confirmation_per_seed.csv']:
            continue
        for row in read_csv(path):
            if row.get('seed') in ['16', '17', '18']: conflicts.append(str(path))
    if conflicts: raise RuntimeError(f'Planned seeds already in historical results: {conflicts[:10]}')
    root.mkdir(parents=True)
    for folder in ['protocol/source_snapshot', 'runs', 'logs', 'summary']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in SOURCES:
        dest = root / 'protocol/source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(BASE / name, dest); hashes[name] = sha(dest)
    shutil.copy2(BASE / 'docs/secondary_cifar100_validation_20261108.md', root / 'protocol/design.md')
    shutil.copy2(BASE / 'docs/secondary_cifar100_run_matrix_20261108.csv', root / 'protocol/run_matrix.csv')
    write_json(root / 'status.json', {'phase': 'preparing', 'completed': 0, 'total': 24, 'updated_at': now()})
    write_json(root / 'protocol/preflight.json', preflight())
    for policy in POLICIES:
        expected, _ = schedule_trace(policy)
        np.save(root / 'protocol' / f'{policy}_expected_update_lr.npy', expected)
    print('Implementation and independent schedule checks passed. Downloading CIFAR-100.', flush=True)
    raw = datasets.CIFAR100(str(data_dir), train=True, download=True)
    assert raw.data.shape == (50000, 32, 32, 3) and len(raw.classes) == 100
    perm = torch.randperm(50000, generator=torch.Generator().manual_seed(1)).tolist()
    split = {'split_seed': 1, 'train_indices': perm[5000:], 'validation_indices': perm[:5000]}
    assert len(set(split['train_indices']) & set(split['validation_indices'])) == 0
    write_json(root / 'protocol/split.json', split)
    sums, squares, count = np.zeros(3), np.zeros(3), 0
    for start in range(0, 45000, 1000):
        block = raw.data[split['train_indices'][start:start+1000]].astype(np.float64) / 255
        sums += block.sum(axis=(0, 1, 2)); squares += (block * block).sum(axis=(0, 1, 2)); count += np.prod(block.shape[:3])
    mean = sums / count; std = np.sqrt(squares / count - mean * mean)
    write_json(root / 'protocol/normalization.json', {'mean': mean.tolist(), 'std': std.tolist(),
               'source': 'frozen 45000 training images only', 'pixel_count_per_channel': int(count)})
    write_json(root / 'protocol/class_counts.json', {key: np.bincount(np.array(raw.targets)[indices], minlength=100).tolist() for key, indices in split.items() if key.endswith('_indices')})
    matrix = read_csv(root / 'protocol/run_matrix.csv')
    assert len(matrix) == 24 and len({(r['scheduler'], r['seed']) for r in matrix}) == 24
    manifest = {'protocol': 'secondary-cifar100-frozen-v1', 'frozen_at': now(), 'data_dir': str(data_dir),
                'policies': POLICIES, 'matrix': matrix, 'source_hashes': hashes,
                'split_sha256': sha(root / 'protocol/split.json'), 'normalization_sha256': sha(root / 'protocol/normalization.json'),
                'torch': torch.__version__, 'torchvision': __import__('torchvision').__version__, 'python': sys.version,
                'gpu': torch.cuda.get_device_name(0), 'timing': 'single process, synchronized first batch to final validation/checkpoint I/O',
                'test_gate': 'all 24 validation-only runs audited and checkpoint hashes locked before test'}
    write_json(root / 'manifest.json', manifest)
    # Actual image smoke checks without using a research seed or evaluating test data.
    torch.set_num_threads(4); set_seed(16000)
    train = data_loader(root, 16000, True); val = data_loader(root, 16000, False)
    x, y = next(iter(train)); assert x.shape == (128, 3, 32, 32) and 0 <= int(y.min()) <= int(y.max()) < 100
    model = build_model('resnet18', num_classes=100).cuda(); criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.0005)
    smoke = []
    for i, (x, y) in enumerate(train):
        opt.zero_grad(set_to_none=True); loss = criterion(model(x.cuda()), y.cuda()); loss.backward(); opt.step()
        assert torch.isfinite(loss); smoke.append(float(loss.detach()))
        if i == 3: break
    model.eval(); x, y = next(iter(val))
    with torch.no_grad(): assert torch.isfinite(criterion(model(x.cuda()), y.cuda()))
    write_json(root / 'protocol/data_smoke.json', {'passed': True, 'train_batches': 4, 'validation_batches': 1,
               'implementation_seed': 16000, 'test_accessed': False, 'candidate_selection': False})
    return manifest


def queue(root, data_dir, resume):
    # A refused invocation does not own an existing campaign and must not rewrite it.
    if not resume and root.exists():
        raise FileExistsError(f'Refusing existing root: {root}')
    try:
        manifest = source_check(root) if resume else prepare(root, data_dir)
        snap = root / 'protocol/source_snapshot'
        for index, row in enumerate(manifest['matrix']):
            policy, seed = row['scheduler'], int(row['seed'])
            dest = root / 'runs' / policy / f'seed{seed}'
            if dest.exists():
                audit(root, policy, seed)
            else:
                source_check(root)
                log = root / 'logs' / f'{index+1:02d}_{policy}_seed{seed}.log'
                command = [sys.executable, '-u', str(snap / 'scripts/run_secondary_cifar100.py'),
                           '--root', str(root), '--run', policy, '--seed', str(seed)]
                with log.open('x', encoding='utf8') as output:
                    child = subprocess.Popen(command, cwd=snap, stdout=output, stderr=subprocess.STDOUT)
                    write_json(root / 'status.json', {'phase': 'training', 'completed': index, 'total': 24,
                               'current_policy': policy, 'seed': seed, 'training_pid': child.pid,
                               'command': command, 'log': str(log), 'updated_at': now()})
                    code = child.wait()
                if code: raise RuntimeError(f'Training failed ({code}), preserved {dest}; see {log}')
                audit(root, policy, seed)
            plots(root, policy, seed)
        locked = []
        for row in manifest['matrix']:
            policy, seed = row['scheduler'], int(row['seed']); c = audit(root, policy, seed)
            locked.append({'policy': policy, 'seed': seed, 'directory': f'runs/{policy}/seed{seed}',
                           'sha256': c['best_sha256'], 'best_epoch': c['best_epoch']})
        lock_path = root / 'selected_checkpoints_locked.json'
        if lock_path.exists(): assert read_json(lock_path)['models'] == locked
        else: write_json(lock_path, {'locked_at': now(), 'models': locked})
        write_json(root / 'status.json', {'phase': 'final_test', 'completed': 24, 'total': 24, 'updated_at': now()})
        evaluate_locked(root); summarize(root)
        write_json(root / 'status.json', {'phase': 'complete', 'completed': 24, 'total': 24, 'updated_at': now()})
    except BaseException as exc:
        if root.exists():
            previous = read_json(root / 'status.json') if (root / 'status.json').exists() else {}
            write_json(root / 'status.json', {**previous, 'phase': 'error', 'error': repr(exc), 'updated_at': now()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=BASE.parent / 'data')
    parser.add_argument('--run', choices=list(POLICIES))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.preflight_only: print(json.dumps(preflight(), indent=2))
    elif args.run: train_run(args.root.resolve(), args.run, args.seed)
    else:
        # OS-released lock survives process crashes without allowing duplicate queues.
        import msvcrt
        lock_path = args.root.resolve().parent / (args.root.name + '.queue.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a+b') as lock_handle:
            if lock_path.stat().st_size == 0:
                lock_handle.write(b'0'); lock_handle.flush()
            lock_handle.seek(0)
            msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
            queue(args.root.resolve(), args.data_dir.resolve(), args.resume)
