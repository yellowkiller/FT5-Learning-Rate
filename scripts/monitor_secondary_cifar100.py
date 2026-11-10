"""Read-only local campaign watchdog; never restarts or modifies training."""
import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def rows(path):
    with Path(path).open(newline='', encoding='utf8') as f:
        return list(csv.DictReader(f))


def normalize(text):
    return str(text).replace('\\', '/').lower()


def processes():
    command = "@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match 'python'} | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                            capture_output=True, text=True, timeout=25,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode: raise RuntimeError('Cannot query Windows processes')
    value = json.loads(result.stdout or '[]')
    if isinstance(value, dict): value = [value]
    return {int(p['ProcessId']): p.get('CommandLine') or '' for p in value}


def tail(path, limit=16000):
    path = Path(path)
    if not path.exists(): return ''
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size-limit))
        return f.read().decode('utf8', errors='replace')


def desktop_notice(body):
    # Local desktop notification only; no email, external messages or app automation.
    script = """Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$notice=New-Object System.Windows.Forms.NotifyIcon
$notice.Icon=[System.Drawing.SystemIcons]::Information
$notice.Visible=$true
$notice.ShowBalloonTip(15000,'FT5 CIFAR-100',BODY,[System.Windows.Forms.ToolTipIcon]::Info)
Start-Sleep -Seconds 16
$notice.Dispose()
""".replace('BODY', "'" + body.replace("'", "''") + "'")
    encoded = base64.b64encode(script.encode('utf-16le')).decode()
    subprocess.Popen(['powershell.exe', '-NoProfile', '-WindowStyle', 'Hidden', '-EncodedCommand', encoded],
                     creationflags=subprocess.CREATE_NO_WINDOW, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def inspect(root, live):
    status = read_json(root / 'status.json')
    process = read_json(root / 'process.json')
    manifest = read_json(root / 'manifest.json')
    progress = read_json(root / 'progress.json') if (root / 'progress.json').exists() else {}
    issues, durations = [], []
    phase = status['phase']
    if phase == 'error': issues.append('Queue error: ' + status.get('error', 'unspecified'))
    if phase not in ['complete', 'paused']:
        cmd = live.get(int(process['queue_pid']), '')
        if 'run_secondary_cifar100' not in cmd or normalize(root) not in normalize(cmd):
            issues.append('Queue PID missing or command line does not match this campaign')
    snapshot = root / 'protocol/source_snapshot'
    for name, expected in manifest['source_hashes'].items():
        if sha(snapshot / name) != expected: issues.append('Frozen source changed: ' + name)
    for name, key in [('split.json', 'split_sha256'), ('normalization.json', 'normalization_sha256')]:
        if sha(root / 'protocol' / name) != manifest[key]: issues.append('Frozen protocol changed: ' + name)
    complete = sorted((root / 'runs').rglob('training_complete.json'))
    for path in complete:
        c = read_json(path); durations.append(c['training_time_seconds'])
        dest = path.parent
        # Hourly checks re-read artifacts: completion JSON alone cannot prove that
        # the CSVs and checkpoints checked last time have remained unchanged.
        epochs = rows(dest / 'epochs.csv')
        lr = rows(dest / 'lr_updates.csv')
        if [int(r['epoch']) for r in epochs] != list(range(1, 151)): issues.append('Incomplete epochs: ' + str(dest))
        if [int(r['update']) for r in lr] != list(range(1, 52801)): issues.append('Incomplete LR log: ' + str(dest))
        if not read_json(dest / 'audit.json').get('passed'): issues.append('Audit failed: ' + str(dest))
        if c['test_acc'] is not None: issues.append('Training record includes test evaluation: ' + str(dest))
        for filename in ['best.pt', 'last.pt']:
            if sha(dest / filename) != c[filename.split('.')[0] + '_sha256']:
                issues.append('Checkpoint changed: ' + str(dest / filename))
        if len(list(dest.glob('*.png'))) < 6: issues.append('Missing run figures: ' + str(dest))
    current_epoch = 0
    if phase == 'training':
        policy, seed = status['current_policy'], int(status['seed'])
        cmd = live.get(int(status.get('training_pid', 0)), '')
        if ('run_secondary_cifar100' not in cmd or normalize(root) not in normalize(cmd)
                or f'--run {policy}' not in cmd or f'--seed {seed}' not in cmd):
            # Re-read status to avoid reporting a normal child-process handover as a failure.
            latest = read_json(root / 'status.json')
            if latest == status: issues.append('Training PID missing or command line mismatch')
        dest = root / 'runs' / policy / f'seed{seed}'
        epoch_file = dest / 'epochs.csv'
        epoch_rows = rows(epoch_file) if epoch_file.exists() else []
        current_epoch = int(epoch_rows[-1]['epoch']) if epoch_rows else 0
        median_seconds = statistics.median([float(r['epoch_seconds']) for r in epoch_rows[-20:]]) if epoch_rows else 30
        start = datetime.fromisoformat(status['updated_at']).timestamp()
        last = max([start] + [p.stat().st_mtime for p in [epoch_file, dest / 'lr_updates.csv'] if p.exists()])
        if time.time() - last > max(600, 15*median_seconds): issues.append('No epoch/update-file progress for over the stall threshold')
        child_log = tail(status['log'])
        if 'Traceback (most recent call last)' in child_log: issues.append('Training traceback in ' + status['log'])
    elif phase == 'final_test':
        evidence = [root / 'status.json'] + list((root / 'runs').rglob('test_result.json'))
        if time.time()-max(p.stat().st_mtime for p in evidence) > 1800:
            issues.append('Final test stage has no new output for 30 minutes')
    stderr = tail(process['stderr'])
    if 'Traceback (most recent call last)' in stderr: issues.append('Queue traceback in ' + process['stderr'])
    if phase not in ['complete', 'final_test'] and list((root / 'runs').rglob('test_result.json')):
        issues.append('Test results appeared before the final-test gate')
    if phase == 'complete':
        if len(complete) != 24 or len(list((root / 'runs').rglob('test_result.json'))) != 24:
            issues.append('Complete status without all 24 training/test results')
        lock = read_json(root / 'selected_checkpoints_locked.json')
        if len(lock['models']) != 24: issues.append('Incomplete checkpoint lock')
        for item in lock['models']:
            dest = root / item['directory']
            if read_json(dest / 'test_result.json')['best_sha256'] != item['sha256']:
                issues.append('Test model does not match locked checkpoint')
        if len(rows(root / 'summary/per_seed.csv')) != 24 or len(rows(root / 'summary/summary.csv')) != 8:
            issues.append('Incomplete final summary')
    remaining = max(0, (24-len(complete))*150-current_epoch)
    estimate = remaining * statistics.median(durations)/150/3600 if durations else None
    return {'checked_at': now(), 'phase': phase, 'completed': len(complete), 'total': 24,
            'current_policy': status.get('current_policy'), 'seed': status.get('seed'), 'epoch': current_epoch,
            'remaining_training_hours_estimate': estimate, 'issues': issues, 'healthy': not issues,
            'queue_log': process['stdout'], 'queue_error_log': process['stderr'], 'last_progress': progress}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=3600)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--notify', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    monitor_dir = root / 'monitor'
    monitor_dir.mkdir(exist_ok=True)
    last_event = None
    while True:
        try:
            report = inspect(root, processes())
        except Exception as exc:
            report = {'checked_at': now(), 'healthy': False, 'phase': 'monitor_error', 'issues': [repr(exc)]}
        # Confirm potential faults after a short grace period for normal handovers,
        # auditing, atomic JSON replacement and per-run plot generation.
        if report['issues']:
            time.sleep(8)
            try:
                report = inspect(root, processes())
            except Exception as exc:
                report = {'checked_at': now(), 'healthy': False, 'phase': 'monitor_error', 'issues': [repr(exc)]}
        report['interval_seconds'] = args.interval
        write_json(monitor_dir / 'latest.json', report)
        event = ('error', tuple(report['issues'])) if report['issues'] else ('complete',) if report['phase'] == 'complete' else None
        if event and event != last_event:
            with (monitor_dir / 'events.jsonl').open('a', encoding='utf8') as f: f.write(json.dumps(report) + '\n')
            write_json(monitor_dir / ('completion.json' if event[0] == 'complete' else 'alert.json'), report)
            if args.notify:
                desktop_notice('All 24 runs and final evaluations completed. Check the summary.' if event[0] == 'complete'
                               else 'Experiment needs attention. Check monitor/alert.json. Training was not restarted.')
            last_event = event
        elif not event:
            last_event = None
        if args.once:
            print(json.dumps(report, indent=2)); break
        if report['phase'] == 'complete' and report['healthy']: break
        time.sleep(max(30, args.interval))


if __name__ == '__main__':
    main()
