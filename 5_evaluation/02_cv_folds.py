"""Folds of the spatial 5-fold cross-validation: every 5 x 5 deg tile goes to one fold.

Folds are balanced by the number of training labels: the tiles in random order (seed 42), each to the fold with the fewest
labels so far. Tiles without labels get a random fold (seed 42).
Input:  WORK/training_table.mat (4_mapping/01_training_table.py)
Output: WORK/cv5/tile_folds.csv (tile, fold, n_labels; = data/evaluation/cv_tile_folds.csv) and WORK/cv5/labels_fold5.mat
        (fold5: fold of every label, in the order of training_table.mat), read by 3_training/run_nn.m (foldvar = fold5).
Next:   3_training/run_nn('networks_cv5.csv') trains one network per fold without the tiles of that fold; 03_cv_auc.py.
Usage:  python 02_cv_folds.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '4_mapping'))     # config.py
from config import WORK, all_tiles, core_tile_of  # noqa: E402

OUT = WORK / 'cv5'
NF = 5


def main():
    OUT.mkdir(exist_ok=True)
    L = sio.loadmat(WORK / 'training_table.mat', variable_names=['lat', 'lon', 'y'])
    tile = core_tile_of(L['lat'].ravel(), L['lon'].ravel())[0]
    n = pd.Series(tile).value_counts()
    rng = np.random.default_rng(42)
    order = rng.permutation(n.index.values)
    load = np.zeros(NF)
    fold = {}
    for t in order:
        k = int(np.argmin(load))
        fold[t] = k
        load[k] += n[t]
    others = sorted(set(all_tiles()) - set(fold))
    for t, k in zip(others, rng.integers(0, NF, len(others))):
        fold[t] = int(k)
    T = pd.DataFrame({'tile': list(fold), 'fold': list(fold.values())})
    T['n_labels'] = T.tile.map(n).fillna(0).astype(int)
    T.sort_values('tile').to_csv(OUT / 'tile_folds.csv', index=False)
    f5 = np.array([fold[t] for t in tile], dtype='int8')
    sio.savemat(OUT / 'labels_fold5.mat', {'fold5': f5.reshape(-1, 1)}, do_compression=False)
    y = L['y'].ravel()
    print('labels per fold:', np.bincount(f5).tolist(),
          'peat share per fold:', [round(float(y[f5 == k].mean()), 3) for k in range(NF)])


if __name__ == '__main__':
    main()
