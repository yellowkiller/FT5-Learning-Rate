"""Fetch exact official CIFAR-100 archive with verified resumable ranges."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / 'tmp/cifar100_download_chunks'
DATA = ROOT / 'data'
URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-100-python.tar.gz'
SIZE = 169001437
MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'
CHUNK = 1024 * 1024


def fetch(index):
    start = index * CHUNK
    end = min(SIZE, start + CHUNK) - 1
    dest = STAGING / f'{index:03d}.part'
    if dest.exists() and dest.stat().st_size == end - start + 1:
        return index
    for attempt in range(5):
        tmp = dest.with_suffix('.downloading')
        header = dest.with_suffix('.headers')
        response = subprocess.run(['curl.exe', '-sS', '--fail', '--location',
            '--connect-timeout', '15', '--max-time', '100', '--range', f'{start}-{end}',
            '--dump-header', str(header), '--output', str(tmp), URL], capture_output=True)
        headers = header.read_text(errors='replace') if header.exists() else ''
        valid_range = re.search(rf'content-range:\s*bytes\s+{start}-{end}/{SIZE}', headers, re.I)
        if response.returncode == 0 and valid_range and tmp.exists() and tmp.stat().st_size == end - start + 1:
            tmp.replace(dest)
            return index
        time.sleep(attempt + 1)
    raise RuntimeError(f'Failed range {start}-{end}: {response.stderr.decode(errors="replace")}')


if __name__ == '__main__':
    STAGING.mkdir(parents=True, exist_ok=True)
    count = (SIZE + CHUNK - 1) // CHUNK
    start_time = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(fetch, i) for i in range(count)]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if done % 8 == 0 or done == count:
                print(f'Verified ranges: {done}/{count}; elapsed {time.monotonic()-start_time:.0f}s', flush=True)
    merged = STAGING / 'cifar-100-python.tar.gz.verified'
    digest = hashlib.md5()
    with merged.open('wb') as out:
        for i in range(count):
            block = (STAGING / f'{i:03d}.part').read_bytes()
            digest.update(block); out.write(block)
    assert merged.stat().st_size == SIZE and digest.hexdigest() == MD5, digest.hexdigest()
    # The failed archive's hash/size is retained in the recovery record.
    target = DATA / 'cifar-100-python.tar.gz'
    old = {'bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()} if target.exists() else None
    (STAGING / 'download_provenance.json').write_text(json.dumps({
        'url': URL, 'bytes': SIZE, 'official_md5': MD5, 'verified': True,
        'replaced_incomplete_archive': old, 'elapsed_seconds': time.monotonic()-start_time}, indent=2))
    merged.replace(target)
    print(f'Official archive verified: {target}; MD5={MD5}', flush=True)
