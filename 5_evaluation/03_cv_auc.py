"""Spatial 5-fold cross-validation AUC of the classification network (18 inputs), by region.

Each fold network (3_training/networks_cv5.csv: cv5_17iv_h10_e800_k0..k4) was trained without the 5 x 5 deg tiles of its
fold (02_cv_folds.py). 200 000 labels per fold (numpy default_rng(7)) are predicted by the network of their fold; AUC =
probability that a random peat label gets a higher PI than a random non-peat label (it does not depend on a threshold).
Regions: Europe; Russia (inside the Russian border of the study-area country layer); Alaska (west of 30 W, lon < -129,
lat > 51; the training labels contain no Canadian pixels); the rest of the labels west of 30 W (contiguous USA, Mexico,
Central America); the map domain (Europe + Russia + Alaska); all labels.
Input:  WORK/training_table.mat, WORK/cv5/labels_fold5.mat, WORK/nn/cv5_17iv_h10_e800_k<k>.mat
Output: results/tables/cv5_auc_regions.csv
Usage:  python 03_cv_auc.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.io as sio
import shapely
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '4_mapping'))     # config.py, matlab_net.py
from config import ROOT, TAB, WORK  # noqa: E402
from matlab_net import MatlabNet  # noqa: E402

CONFIG = 'cv5_17iv_h10_e800'
COLS = list(range(17)) + [17]                    # 17 inputs + OPERA inundated vegetation
COUNTRIES = ROOT / 'Raw/Study_Area/Pan_boreal/Northern_study_area_country.shp'


def main():
    L = sio.loadmat(WORK / 'training_table.mat', variable_names=['X', 'y', 'lat', 'lon'])
    f5 = sio.loadmat(WORK / 'cv5/labels_fold5.mat')['fold5'].ravel()
    rng = np.random.default_rng(7)
    idx = np.concatenate([rng.choice(np.flatnonzero(f5 == k), 200_000, replace=False) for k in range(5)])
    X, fold = L['X'][idx], f5[idx]
    lat, lon, y = L['lat'].ravel()[idx], L['lon'].ravel()[idx], L['y'].ravel()[idx].astype(int)

    # out-of-fold PI
    p = np.full(len(idx), np.nan, 'float32')
    for k in range(5):
        net = MatlabNet(WORK / 'nn' / f'{CONFIG}_k{k}.mat')
        s = fold == k
        x = X[s][:, COLS]
        ok = np.isfinite(x).all(1)
        pk = np.full(int(s.sum()), np.nan, 'float32')
        pk[ok] = net.predict(x[ok])
        p[s] = pk

    C = gpd.read_file(COUNTRIES)
    ru = C[C.ADMIN == 'Russia'].union_all()
    shapely.prepare(ru)
    west = lon < -30
    rus = ~west & shapely.contains_xy(ru, lon, lat)
    alaska = west & (lon < -129) & (lat > 51)
    regions = {'all': np.ones(len(y), bool), 'Europe': ~west & ~rus, 'Russia': rus, 'Alaska': alaska,
               'west of 30 W outside Alaska': west & ~alaska, 'map domain (Europe + Russia + Alaska)': ~west | alaska}
    rows = []
    for reg, m in regions.items():
        s = m & np.isfinite(p)
        tiles = len(set(zip(np.floor(lat[s] / 5), np.floor(lon[s] / 5))))
        rows.append({'region': reg, 'n': int(s.sum()), 'tiles': tiles, 'peat_share': round(float(y[s].mean()), 3),
                     'AUC': roc_auc_score(y[s], p[s])})
    R = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    R.to_csv(TAB / 'cv5_auc_regions.csv', index=False)
    print(R.to_string(index=False, float_format='%.4f'))


if __name__ == '__main__':
    main()
