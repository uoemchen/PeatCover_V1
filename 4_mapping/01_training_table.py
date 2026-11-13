"""Training table of the classification network: labels + 17 inputs + OPERA, one file for MATLAB and Python.

Input : the training labels of 2_labels/ (sample_split_labels.py): 10 M pixels with the 17 inputs, peat / non-peat labels
        from the agreement of the a priori inventories, 1:1, random 70 / 15 / 15 % split
Adds  : opera_iv, opera_ow = OPERA occurrence (%) of the same 90 m cell (1_inputs/opera_90m.py); fold = spatial block (2 x 4 deg)
Output: WORK/training_table.mat (X: N x 19 = 17 inputs + 2 OPERA layers; y; split 0 train / 1 validation / 2 test;
        fold; lat; lon). 3_training/run_nn.m selects the inputs: feats '17' (17 inputs) or '17iv' (17 + IV = 18 inputs).
Usage: python 01_training_table.py
"""
import time

import numpy as np
import scipy.io as sio

from config import FEATURES17, LABELS, OPERA_NAMES, WORK, block_fold, opera_at

if __name__ == '__main__':
    t0 = time.time()
    parts = []
    for s, name in enumerate(['train', 'validation', 'test']):
        m = sio.loadmat(LABELS / f'{name}_trainingData.mat', variable_names=['input', 'target', 'latS', 'lonS'])
        parts.append((m['input'].astype('float32'), m['target'].ravel().astype('int8'),
                      np.full(len(m['target']), s, 'int8'), m['latS'].ravel(), m['lonS'].ravel()))
        print(name, len(m['target']), f'{time.time() - t0:.0f}s', flush=True)
    X17 = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    split = np.concatenate([p[2] for p in parts])
    lat = np.concatenate([p[3] for p in parts]).astype('float32')
    lon = np.concatenate([p[4] for p in parts]).astype('float32')
    op = opera_at(lat, lon)
    print('OPERA missing share', np.isnan(op).any(1).mean().round(5), f'{time.time() - t0:.0f}s', flush=True)
    X = np.hstack([X17, op]).astype('float32')
    fold = block_fold(lat, lon).astype('int8')
    sio.savemat(WORK / 'training_table.mat',
                {'X': X, 'y': y, 'split': split, 'fold': fold, 'lat': lat, 'lon': lon,
                 'features': np.array(FEATURES17 + OPERA_NAMES, dtype=object)}, do_compression=False)
    print('N', len(y), 'peat share', y.mean().round(4), 'fold sizes', np.bincount(fold),
          'peat share by fold', [round(float(y[fold == k].mean()), 3) for k in range(5)], f'{time.time() - t0:.0f}s')
