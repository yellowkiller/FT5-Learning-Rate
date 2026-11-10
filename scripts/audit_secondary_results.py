"""Audit saved CIFAR-100 records without loading images or evaluating models.

The default mode works on a Git checkout without local checkpoints. Add
--verify-local-checkpoints to hash the original model files as well.
Only the separate catalog directory is written; campaign records are read-only.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / 'results/secondary_validation/20261108_cifar100_frozen_v1_download_retry1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, context):
    if not condition:
        raise ValueError(context)


def close(actual, expected, context):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), expected,
            rel_tol=1e-10, abs_tol=1e-9), context)


def expected_lr(policy, update):
    """Independent fixed-protocol formula, indexed before optimizer.step()."""
    epoch = update // 352
    if policy == 'constant': return .1
    if policy == 'step': return .1 * .1 ** (epoch // 50)
    if policy == 'multistep': return .1 * .1 ** ((epoch >= 75) + (epoch >= 112))
    if policy == 'cosine': return .05 * (1 + math.cos(math.pi * epoch / 150))
    if policy == 'warmrestart':
        period, position = (50, epoch) if epoch < 50 else (100, epoch - 50)
        return .05 * (1 + math.cos(math.pi * position / period))
    if policy == 'warmup_cosine':
        return .1 * (.01 + .99 * epoch / 7) if epoch < 7 else .05 * (1 + math.cos(math.pi * (epoch - 7) / 143))
    if policy == 'cyclic':
        cycle = math.floor(1 + update / 1408)
        return .01 + .09 * max(0, 1 - abs(update / 704 - 2 * cycle + 1)) / 2 ** (cycle - 1)
    require(policy == 'onecycle', 'Unknown policy: ' + policy)
    peak, start, minimum = .3 * 52800 - 1, .03 / 25, .03 / 25 / 10000
    if update <= peak:
        return .03 + (start - .03) * (math.cos(math.pi * update / peak) + 1) / 2
    return minimum + (.03 - minimum) * (math.cos(math.pi * (update - peak) / (52799 - peak)) + 1) / 2


def measurements(epoch_rows, complete, test):
    vals = [float(row['val_acc']) * 100 for row in epoch_rows]
    result = {'best_validation_pct': complete['best_val_acc'] * 100,
              'final_validation_pct': vals[-1], 'test_pct': test['test_acc'] * 100,
              'late_validation_sd_pp': statistics.stdev(vals[-10:]),
              'auc_pct': sum((a+b)/2 for a,b in zip(vals, vals[1:]))/149,
              'auc30_pct': sum((a+b)/2 for a,b in zip(vals[:29], vals[1:30]))/29,
              'training_minutes': complete['training_time_seconds']/60}
    for threshold in [50, 60, 70]:
        first = next((i+1 for i,v in enumerate(vals) if v >= threshold), None)
        sustained = next((i+1 for i in range(146) if min(vals[i:i+5]) >= threshold), None)
        terminal = next((i+1 for i in range(150) if min(vals[i:]) >= threshold), None)
        for kind, value in [('first', first), ('sustained', sustained), ('terminal', terminal)]:
            result[f'{kind}{threshold}_epoch'] = value
            result[f'{kind}{threshold}_updates'] = value*352 if value is not None else None
        result[f'sustained{threshold}_confirmed_epoch'] = sustained+4 if sustained is not None else None
    return result


def audit(root, output, verify_checkpoints):
    require(output != root and root not in output.parents, 'Write audit outside the campaign root')
    manifest = read(root/'manifest.json')
    policies = list(manifest['policies'])
    expected_keys = {(policy, seed) for policy in policies for seed in [16,17,18]}
    require(len(policies) == 8 and read(root/'status.json')['phase'] == 'complete', 'Campaign incomplete')
    matrix_keys = {(r['scheduler'], int(r['seed'])) for r in manifest['matrix']}
    require(matrix_keys == expected_keys and len(manifest['matrix']) == 24, 'Run matrix mismatch')
    for name, digest in manifest['source_hashes'].items():
        require(sha(root/'protocol/source_snapshot'/name) == digest, 'Frozen source changed: '+name)
    for name, key in [('split.json','split_sha256'), ('normalization.json','normalization_sha256')]:
        require(sha(root/'protocol'/name) == manifest[key], 'Frozen data protocol changed: '+name)
    split = read(root/'protocol/split.json')
    train, val = set(split['train_indices']), set(split['validation_indices'])
    require(len(train)==45000 and len(val)==5000 and not train & val and train | val == set(range(50000)), 'Invalid split')
    lock = read(root/'selected_checkpoints_locked.json')
    locked = {(r['policy'],int(r['seed'])):r for r in lock['models']}
    require(len(lock['models']) == 24 and set(locked)==expected_keys, 'Invalid checkpoint lock')
    locked_at = datetime.fromisoformat(lock['locked_at'])
    per_seed = {(r['policy'],int(r['seed'])):r for r in rows(root/'summary/per_seed.csv')}
    require(len(rows(root/'summary/per_seed.csv')) == 24 and set(per_seed)==expected_keys, 'Per-seed table mismatch')
    computed, artifacts, checkpoint_count, lr_count = {}, [], 0, 0
    max_error = {policy:0. for policy in policies}
    for policy, seed in sorted(expected_keys):
        dest = root/f'runs/{policy}/seed{seed}'
        complete, test, meta = [read(dest/name) for name in ['training_complete.json','test_result.json','metadata.json']]
        epoch_rows = rows(dest/'epochs.csv')
        require([int(r['epoch']) for r in epoch_rows] == list(range(1,151)), f'Epoch sequence: {dest}')
        require(complete['policy']==policy and complete['seed']==seed and complete['epochs']==150, 'Run identity mismatch')
        require(meta['dataset']=='cifar100' and meta['classes']==100 and meta['seed']==seed and meta['policy']==policy, 'Metadata mismatch')
        require(meta['split_sha256']==manifest['split_sha256'] and meta['normalization_sha256']==manifest['normalization_sha256'], 'Data hashes differ')
        require(meta['source_hashes']==manifest['source_hashes'] and meta['test_evaluation'] is False and complete['test_acc'] is None, 'Training record includes test access or changed source')
        for row in epoch_rows:
            require(all(math.isfinite(float(v)) for v in row.values()), 'Nonfinite epoch value')
        selected = min(epoch_rows, key=lambda r:(-float(r['val_acc']),float(r['val_loss']),int(r['epoch'])))
        require(int(selected['epoch'])==complete['best_epoch']==locked[policy,seed]['best_epoch'], 'Checkpoint selection mismatch')
        close(selected['val_acc'],complete['best_val_acc'],'Best accuracy mismatch')
        close(selected['val_loss'],complete['best_val_loss'],'Best loss mismatch')
        require(locked[policy,seed]['sha256']==complete['best_sha256']==test['best_sha256'], 'Locked/test hash mismatch')
        require(datetime.fromisoformat(complete['finished_at']) <= locked_at <= datetime.fromisoformat(test['evaluated_at']), 'Test/lock chronology mismatch')
        require(read(dest/'audit.json')['passed'], 'Original audit failed')
        require(len(list(dest.glob('*.png'))) >= 6, 'Missing figures')
        n = 0
        with (dest/'lr_updates.csv').open(newline='',encoding='utf8') as handle:
            for i,row in enumerate(csv.DictReader(handle)):
                require(int(row['update'])==i+1 and int(row['epoch'])==i//352+1 and int(row['batch'])==i%352+1, 'LR sequence mismatch')
                require(float(row['momentum'])==.9, 'Momentum changed')
                error=abs(float(row['lr'])-expected_lr(policy,i))
                max_error[policy]=max(max_error[policy],error)
                require(error < 1e-10, 'LR formula mismatch: '+str(dest))
                n+=1
        require(n==52800, 'LR log incomplete')
        lr_count+=n
        measured=measurements(epoch_rows,complete,test)
        for key,value in measured.items():
            actual=per_seed[policy,seed][key]
            if value is None: require(actual=='', 'Missing threshold recorded as a value')
            else: close(actual,value,f'{policy}/{seed} {key}')
        computed[policy,seed]=measured
        if verify_checkpoints:
            for filename,key in [('best.pt','best_sha256'),('last.pt','last_sha256')]:
                require(sha(dest/filename)==complete[key], 'Local checkpoint changed: '+str(dest/filename))
                checkpoint_count+=1
    summary=rows(root/'summary/summary.csv')
    require(len(summary)==8 and {r['policy'] for r in summary}==set(policies), 'Summary policy mismatch')
    for row in summary:
        group=[computed[row['policy'],seed] for seed in [16,17,18]]
        for key in group[0]:
            values=[r[key] for r in group if r[key] is not None]
            require(int(row[key+'_n'])==len(values), 'Attainment count mismatch')
            for suffix,value in [('mean',statistics.mean(values) if values else None),('sample_sd',statistics.stdev(values) if len(values)>1 else None)]:
                if value is None:require(row[key+'_'+suffix]=='','Unexpected summary value')
                else:close(row[key+'_'+suffix],value,'Summary '+row['policy']+' '+key+' '+suffix)
    pairs=rows(root/'summary/paired_differences.csv')
    require(len(pairs)==15, 'Missing paired differences')
    for row in pairs:
        a,b=[computed[policy,int(row['seed'])] for policy in [row['left'],row['right']]]
        for key in a:
            value=a[key]-b[key] if a[key] is not None and b[key] is not None else None
            if value is None: require(row[key+'_delta']=='','Unexpected paired threshold')
            else:close(row[key+'_delta'],value,'Paired difference '+key)
    # Inventory original artifacts, including the locally retained recovery backup.
    # Hash model files only when explicitly requested; summaries already record hashes.
    for path in sorted(root.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:continue
        local = path.suffix in ['.pt','.log','.tmp'] or 'process' in path.name or 'monitor' in path.relative_to(root).parts
        digest=sha(path) if path.suffix!='.pt' or verify_checkpoints else ''
        artifacts.append({'path':path.relative_to(PROJECT).as_posix() if path.is_relative_to(PROJECT) else str(path),
                          'bytes':path.stat().st_size,'sha256':digest,'storage':'local_only' if local else 'git_record'})
    report={'passed':True,'campaign':root.relative_to(PROJECT).as_posix() if root.is_relative_to(PROJECT) else str(root),
            'training_runs':24,'test_records':24,'epoch_rows':3600,'lr_update_rows':lr_count,
            'local_checkpoints_verified':checkpoint_count,'checkpoint_verification_requested':verify_checkpoints,
            'source_hashes_verified':True,'split_and_normalization_verified':True,
            'validation_selection_and_test_gate_verified':True,'all_saved_summary_values_recomputed':True,
            'analytic_lr_max_abs_error':max_error,'training_or_test_image_evaluation_performed':False}
    output.mkdir(parents=True,exist_ok=True)
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    with (output/'artifact_manifest.csv').open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(artifacts[0]));writer.writeheader();writer.writerows(artifacts)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--output',type=Path,default=PROJECT/'results/secondary_validation/catalog')
    parser.add_argument('--verify-local-checkpoints',action='store_true')
    args=parser.parse_args()
    audit(args.root.resolve(),args.output.resolve(),args.verify_local_checkpoints)
