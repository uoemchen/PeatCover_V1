"""North America: the PI threshold of northern Canada, transferred from Eurasia.

Northern Canada (outside the PEATMAP Canada polygons and outside the tiles with a reference fraction) has no reference peat
fraction. A pixel is peat there if its classification PI >= T, with T transferred from the Eurasian tiles N65E055-N65E065
(65-70 N, 55-70 E): T = the k-th largest PI of these tiles, k = the number of peat pixels of the Eurasian map (03) there.
For the 18-input model T = 82 (NORTH in 06_north_america_pi.py).
Output: results/tables/na_transfer_threshold.csv
Usage: python 07_north_threshold.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

from config import ROOT, TAB, WORK  # noqa: E402

PO = ROOT / 'Prediction_Output'
EU = PO / 'PeatCover_submitted'
CMP = WORK / 'eurasia_cmp'
EU_COVER = EU / 'PeatCover_submitted_Eurasia.tif'
EU_PI = EU / 'PI_Eurasia.tif'
TILES = ('N65E055', 'N65E060', 'N65E065')                # 65-70 N, 55-70 E


def kth_threshold(pi, cover, ok):
    n = int(((cover == 1) & ok).sum())
    return n / ok.sum(), np.sort(pi[ok])[::-1][n - 1]


def transfer_threshold():
    rows = []
    b = (55, 65, 70, 70)
    A = {}
    for k, f in (('cover', EU_COVER), ('pi', EU_PI)):
        with rasterio.open(f) as r:
            A[k] = r.read(1, window=from_bounds(*b, transform=r.transform).round_offsets().round_lengths())
            A[k + '_nd'] = r.nodata
    share, thr = kth_threshold(A['pi'], A['cover'], (A['cover'] != A['cover_nd']) & (A['pi'] != A['pi_nd']))
    rows.append({'map': 'published (V2)', 'peat share (%)': 100 * share, 'threshold (PI x 100)': int(thr)})
    for v, name in (('v17', '17 inputs (03_eurasia_map)'), ('v17iv', '18 inputs (03_eurasia_map)')):
        pis, cov = [], []
        for t in TILES:
            with rasterio.open(CMP / v / 'pi_u8' / f'{t}.tif') as r:
                p = r.read(1)
            with rasterio.open(CMP / v / 'cover' / f'{t}.tif') as r:
                c = r.read(1)
            ok = (c != 255) & (p != 255)
            pis.append(p[ok])
            cov.append(c[ok])
        p, c = np.concatenate(pis), np.concatenate(cov)
        share, thr = kth_threshold(p, c, np.ones(len(p), bool))
        rows.append({'map': name, 'peat share (%)': 100 * share, 'threshold (PI x 100)': int(thr)})
    D = pd.DataFrame(rows)
    D.round(2).to_csv(TAB / 'na_transfer_threshold.csv', index=False)
    print(D.round(2).to_string(index=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    transfer_threshold()
