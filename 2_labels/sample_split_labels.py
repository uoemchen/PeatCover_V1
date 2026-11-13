"""Draw the training labels from the per-tile training data and split them into train / validation / test sets.

10 million pixels: half from the tiles whose centre is inside the CORINE coverage, half from the other tiles; in each
half, 50 % peat and 50 % non-peat. Each tile contributes in proportion to its number of peat (non-peat) pixels. The pixels
are then shuffled and split 70 / 15 / 15 %.
Input:  <data_folder>/<tile>_trainingData.mat (BuildTrainingData.m), Raw/Study_Area/CORINE_coverage.shp
Output: <output_folder>/train_trainingData.mat, validation_trainingData.mat, test_trainingData.mat
        (input: n x 17, target: n x 1, latS / lonS) and global_sampling_statistics.json
The tiles are sampled in parallel worker processes, so a new run gives another draw with the same design.
"""
import gc
import json
import os
import random
import re
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import scipy.io
from shapely.geometry import Point
from shapely.strtree import STRtree

ROOT = Path(os.environ.get('PEATCOVER_ROOT', 'path/to/PEATLAND'))
DATA_FOLDER = ROOT / 'Training_database/training_data_tiles'
OUTPUT_FOLDER = ROOT / 'Training_database/training_labels'
CORINE_COVERAGE = ROOT / 'Raw/Study_Area/CORINE_coverage.shp'

TOTAL_SAMPLES = 10_000_000
CORINE_RATIO = 0.5
PEAT_RATIO = 0.5
SPLITS = {'train': 0.7, 'val': 0.15, 'test': 0.15}
RANDOM_SEED = 42
N_WORKERS = 12


def tile_center(filename, tile_size=5.0):
    """centre of a tile from its name (lower-left corner, e.g. N60E025)"""
    ns, lat, ew, lon = re.match(r'([NS])(\d+)([EW])(\d+)', filename.replace('_trainingData.mat', '')).groups()
    lat = -float(lat) if ns == 'S' else float(lat)
    lon = -float(lon) if ew == 'W' else float(lon)
    return lat + tile_size / 2.0, lon + tile_size / 2.0


def tile_counts(mat_file):
    """number of peat and non-peat pixels of a tile"""
    try:
        with h5py.File(str(mat_file), 'r') as f:
            target = np.array(f['target']).flatten()
        return mat_file.stem.replace('_trainingData', ''), {
            'file_path': mat_file, 'peat_count': int(np.sum(target == 1)), 'nonpeat_count': int(np.sum(target == 0))}
    except Exception as e:
        print(f'Cannot read {mat_file}: {e}')
        return None


def sample_tile(mat_file, peat_quota, nonpeat_quota):
    """random peat and non-peat pixels of one tile"""
    samples = {'peat': [], 'nonpeat': []}
    if peat_quota == 0 and nonpeat_quota == 0:
        return samples
    with h5py.File(str(mat_file), 'r') as f:
        target = f['target'][()].flatten()
        lat = f['latS'][()].flatten()
        lon = f['lonS'][()].flatten()
        features = f['input'][()]
    if features.shape[0] < features.shape[1] and features.shape[0] <= 1000:     # MATLAB v7.3 files are transposed
        features = features.T
    tile_name = mat_file.stem.replace('_trainingData', '')

    for key, label, quota in (('peat', 1, peat_quota), ('nonpeat', 0, nonpeat_quota)):
        idx_all = np.where(target == label)[0]
        if quota > 0 and len(idx_all) > 0:
            for idx in np.random.choice(idx_all, size=min(quota, len(idx_all)), replace=False):
                samples[key].append({'features': features[idx].copy(), 'label': label, 'tile_name': tile_name,
                                     'latitude': float(lat[idx]), 'longitude': float(lon[idx])})
    return samples


def sample_region(tiles, peat_quota, nonpeat_quota, name):
    with Pool(processes=N_WORKERS) as pool:
        stats = dict(r for r in pool.map(tile_counts, tiles) if r is not None)
    total_peat = sum(s['peat_count'] for s in stats.values())
    total_nonpeat = sum(s['nonpeat_count'] for s in stats.values())
    print(f'{name}: {len(stats)} tiles, {total_peat:,} peat and {total_nonpeat:,} non-peat pixels')

    tasks = []
    for s in stats.values():
        p = int(peat_quota * s['peat_count'] / total_peat) if total_peat > 0 else 0
        n = int(nonpeat_quota * s['nonpeat_count'] / total_nonpeat) if total_nonpeat > 0 else 0
        tasks.append((s['file_path'], min(p, s['peat_count']), min(n, s['nonpeat_count'])))
    with Pool(processes=N_WORKERS) as pool:
        results = pool.starmap(sample_tile, tasks)
    peat = [x for r in results for x in r['peat']]
    nonpeat = [x for r in results for x in r['nonpeat']]
    print(f'{name}: sampled {len(peat):,} peat and {len(nonpeat):,} non-peat pixels')
    return peat, nonpeat


def save_split(name, samples):
    n = len(samples)
    input_matrix = np.zeros((n, len(samples[0]['features'])), dtype=np.float32)
    target = np.zeros((n, 1), dtype=np.float32)
    lat = np.zeros((n, 1), dtype=np.float32)
    lon = np.zeros((n, 1), dtype=np.float32)
    for i, s in enumerate(samples):
        input_matrix[i, :] = s['features']
        target[i, 0] = s['label']
        lat[i, 0] = s['latitude']
        lon[i, 0] = s['longitude']
    peat = int(np.sum(target == 1))
    metadata = {'total_samples': n, 'peat_samples': peat, 'nonpeat_samples': n - peat,
                'feature_count': input_matrix.shape[1], 'has_coordinates': True,
                'peat_ratio': float(peat / n) if n > 0 else 0,
                'tiles_included': list(set(s['tile_name'] for s in samples))}
    filename = 'validation_trainingData.mat' if name == 'val' else f'{name}_trainingData.mat'
    scipy.io.savemat(str(OUTPUT_FOLDER / filename),
                     {'input': input_matrix, 'target': target, 'latS': lat, 'lonS': lon, 'metadata': metadata},
                     format='5', do_compression=True)
    print(f'{filename}: {n:,} pixels ({peat:,} peat)')
    return {'total_samples': n, 'peat_samples': peat, 'nonpeat_samples': n - peat,
            'peat_ratio': peat / n if n > 0 else 0}


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    corine_samples = int(TOTAL_SAMPLES * CORINE_RATIO)
    non_corine_samples = TOTAL_SAMPLES - corine_samples
    quota = {'corine_peat': int(corine_samples * PEAT_RATIO),
             'corine_nonpeat': corine_samples - int(corine_samples * PEAT_RATIO),
             'non_corine_peat': int(non_corine_samples * PEAT_RATIO),
             'non_corine_nonpeat': non_corine_samples - int(non_corine_samples * PEAT_RATIO)}

    # tiles whose centre is inside the CORINE coverage
    coverage = gpd.read_file(CORINE_COVERAGE)
    if coverage.crs != 'EPSG:4326':
        coverage = coverage.to_crs('EPSG:4326')
    tree = STRtree(coverage.geometry.tolist())

    def in_corine(lat, lon):
        point = Point(lon, lat)
        return any(coverage.geometry.iloc[i].contains(point) for i in tree.query(point))

    mat_files = sorted(DATA_FOLDER.glob('*_trainingData.mat'))
    corine_tiles = [f for f in mat_files if in_corine(*tile_center(f.name))]
    other_tiles = [f for f in mat_files if f not in corine_tiles]
    print(f'{len(corine_tiles)} CORINE tiles, {len(other_tiles)} other tiles')

    peat_c, nonpeat_c = sample_region(corine_tiles, quota['corine_peat'], quota['corine_nonpeat'], 'CORINE')
    peat_o, nonpeat_o = sample_region(other_tiles, quota['non_corine_peat'], quota['non_corine_nonpeat'], 'other')

    # shuffle and split 70 / 15 / 15
    all_samples = peat_c + peat_o + nonpeat_c + nonpeat_o
    random.shuffle(all_samples)
    n_train = int(len(all_samples) * SPLITS['train'])
    n_val = int(len(all_samples) * SPLITS['val'])
    splits = {'train': all_samples[:n_train], 'val': all_samples[n_train:n_train + n_val],
              'test': all_samples[n_train + n_val:]}

    results = {}
    for name, samples in splits.items():
        results[name] = save_split(name, samples)
        gc.collect()

    config = {'total_samples': TOTAL_SAMPLES, 'corine_samples': corine_samples,
              'non_corine_samples': non_corine_samples, **quota, 'splits': SPLITS, 'n_workers': N_WORKERS}
    with open(OUTPUT_FOLDER / 'global_sampling_statistics.json', 'w') as f:
        json.dump({'sampling_config': config, 'actual_results': results}, f, indent=2)


if __name__ == '__main__':
    main()
