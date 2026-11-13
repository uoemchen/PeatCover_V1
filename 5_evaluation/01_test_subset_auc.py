"""AUC and accuracy (PI >= 50) of the trained networks on the random test subset of the training labels.

The 1.5 M test pixels of the random split (4_mapping/01_training_table.py, split = 2), for every network in WORK/nn/
(3_training/networks_final.csv). AUC does not depend on a threshold.
Output: results/tables/label_metrics.csv
Usage: python 01_test_subset_auc.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '4_mapping'))     # config.py, matlab_net.py
from config import TAB, WORK, input_cols
from matlab_net import MatlabNet


if __name__ == '__main__':
    D = sio.loadmat(WORK / 'training_table.mat', variable_names=['X', 'y', 'split', 'fold'])
    X, y, split, fold = D['X'], D['y'].ravel(), D['split'].ravel(), D['fold'].ravel()
    ok = np.isfinite(X).all(1)
    rows = []
    te = np.where((split == 2) & ok)[0]
    for f in sorted((WORK / 'nn').glob('*.mat')):
        name = f.stem
        if name.endswith('_check') or name.startswith('cv'):
            continue
        p = MatlabNet(f).predict(X[te][:, input_cols(name)])
        rows.append({'model': name, 'check': 'random test', 'n': len(te), 'AUC': roc_auc_score(y[te], p),
                     'acc@0.5': ((p >= 0.5) == y[te]).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / 'label_metrics.csv', index=False)
    print(out.round(4).to_string(index=False))
