"""Build the photo caches once (resumable).

  cache/base_SIZE.npy     every photo (train, val, test, inadequate), padded to a square and resized, uint8
  cache/test_SIZE.npy     every test photo under every test condition (pre-degraded, so all models see
                          exactly the same degraded pixels)
  cache/quality_SIZE.csv  q_lap / q_oracle for every clean photo and every test photo x condition
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import common as C
import degrade

_BASE = None


def pad_resize(image_id):
    with Image.open(C.IMAGES / f'{image_id}.jpg') as source:
        image = ImageOps.pad(source.convert('RGB'), (C.IMAGE_SIZE, C.IMAGE_SIZE),
                             method=Image.Resampling.BICUBIC, color=(0, 0, 0))
        pixels = np.asarray(image, dtype=np.uint8)
    return pixels, degrade.q_lap(pixels)


def _open_base(path):
    global _BASE
    _BASE = np.load(path, mmap_mode='r')


def degrade_row(job):
    row, image_id, condition = job
    kind, value = degrade.parse(condition)
    pixels = degrade.apply(np.array(_BASE[row]), kind, value, seed=C.stable_seed(image_id, condition))
    return pixels, degrade.q_lap(pixels), degrade.severity(kind, value)


def cache_settings():
    """Everything the cached pixels depend on besides the photo list."""
    import hashlib
    return {'size': C.IMAGE_SIZE, 'conditions': C.TEST_CONDITIONS,
            'degrade_py': hashlib.sha256((C.ROOT / 'src' / 'degrade.py').read_bytes()).hexdigest()[:16]}


FORCE = False


def up_to_date(meta_path, ids):
    if FORCE or not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text())
    return meta.get('ids') == ids and meta.get('settings') == cache_settings()


def build(workers):
    C.CACHE.mkdir(parents=True, exist_ok=True)
    table = C.load_table()
    ids = table.image_id.tolist()
    base_path, base_meta = C.CACHE / f'base_{C.IMAGE_SIZE}.npy', C.CACHE / f'base_{C.IMAGE_SIZE}.json'
    test_path, test_meta = C.CACHE / f'test_{C.IMAGE_SIZE}.npy', C.CACHE / f'test_{C.IMAGE_SIZE}.json'
    quality_path = C.CACHE / f'quality_{C.IMAGE_SIZE}.csv'

    if up_to_date(base_meta, ids) and quality_path.is_file():
        print(f'base cache up to date ({len(ids)} photos)', flush=True)
        clean_q = pd.read_csv(quality_path, dtype={'image_id': str})
        clean_q = clean_q[clean_q.condition == 'clean'].set_index('image_id').q_lap.reindex(ids).to_numpy()
    else:
        tmp = C.CACHE / f'.base_{C.IMAGE_SIZE}.npy.tmp'
        array = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(len(ids), C.IMAGE_SIZE, C.IMAGE_SIZE, 3))
        clean_q = np.zeros(len(ids))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, (pixels, q) in enumerate(pool.map(pad_resize, ids, chunksize=16)):
                array[i], clean_q[i] = pixels, q
                if (i + 1) % 500 == 0:
                    print(f'base cache: {i + 1}/{len(ids)}', flush=True)
        array.flush()
        del array
        tmp.replace(base_path)
        base_meta.write_text(json.dumps({'size': C.IMAGE_SIZE, 'ids': ids, 'settings': cache_settings()}) + '\n')
        print(f'base cache: {len(ids)} photos at {C.IMAGE_SIZE}px', flush=True)

    test_ids = table.loc[table.split == 'test', 'image_id'].tolist()
    keys = [[i, c] for i in test_ids for c in C.TEST_CONDITIONS]
    quality = [{'image_id': i, 'condition': 'clean', 'q_lap': q, 'q_oracle': 0.0} for i, q in zip(ids, clean_q)]
    if up_to_date(test_meta, keys) and quality_path.is_file():
        print(f'test cache up to date ({len(keys)} photo x condition rows)', flush=True)
        return
    row_of = {i: r for r, i in enumerate(ids)}
    jobs = [(row_of[i], i, c) for i, c in keys]
    tmp = C.CACHE / f'.test_{C.IMAGE_SIZE}.npy.tmp'
    array = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(len(jobs), C.IMAGE_SIZE, C.IMAGE_SIZE, 3))
    with ProcessPoolExecutor(max_workers=workers, initializer=_open_base, initargs=(str(base_path),)) as pool:
        for n, (pixels, q, sev) in enumerate(pool.map(degrade_row, jobs, chunksize=16)):
            array[n] = pixels
            _, image_id, condition = jobs[n]
            if condition != 'clean':
                quality.append({'image_id': image_id, 'condition': condition, 'q_lap': q, 'q_oracle': sev})
            if (n + 1) % 1000 == 0:
                print(f'test cache: {n + 1}/{len(jobs)}', flush=True)
    array.flush()
    del array
    tmp.replace(test_path)
    pd.DataFrame(quality).to_csv(quality_path, index=False)
    test_meta.write_text(json.dumps({'size': C.IMAGE_SIZE, 'ids': keys, 'settings': cache_settings()}) + '\n')
    print(f'test cache: {len(test_ids)} photos x {len(C.TEST_CONDITIONS)} conditions', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--force', action='store_true', help='Rebuild even if the cache looks up to date.')
    args = parser.parse_args()
    global FORCE
    FORCE = args.force
    build(max(1, args.workers))


if __name__ == '__main__':
    main()
