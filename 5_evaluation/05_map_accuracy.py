"""Evaluation of the PeatCover map with independent field observations (evaluation set 4, Table 3 of the paper).

Input : data/evaluation/evaluation_points_set4.csv.gz, one row per 90 m pixel (the UK LandIS augers of set 4 are not
        included: their licence does not allow redistribution).
        Map values: by default those stored in the file (map_submitted, map_revised); with --map, the given GeoTIFF
        (e.g. PeatCover.tif from Zenodo: 0 = non-peat, 1 = peat, 255 = no data) is read at the pixel centres instead of
        map_revised.
Metrics: recall of peat, recall of non-peat, balanced accuracy (BA, their mean), overall accuracy, precision and F1 of
peat; weighted by pixels ('points') and by 0.5 x 0.5 deg blocks ('blocks': every block counts once per class); 95 %
interval of BA from a bootstrap of the blocks (1000 replicates). A recall needs >= 30 pixels of its class.
As in the paper, only the pixels with a value on all the maps compared there are used (used_in_table3 = 1; --all-pixels
uses every pixel with a value on the maps evaluated here).
Groups: all regions, Eurasia, Europe (excl. Russia), Russia, North America, United Kingdom, the regions, the datasets.
Output: results/evaluation_metrics.csv (one row per group, weighting and map), results/evaluation_difference.csv
        (revised - submitted, with 95 % intervals)
Usage: python 05_map_accuracy.py [--points <csv>] [--map PeatCover.tif] [--all-pixels]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
POINTS = HERE.parent / 'data/evaluation/evaluation_points_set4.csv.gz'
OUT = HERE.parent / 'results'
UK = ['UK: Scotland', 'UK: England & Wales', 'UK: Northern Ireland']
REGIONS = UK + ['Ireland', 'Netherlands', 'Germany', 'Poland', 'Other Europe', 'Baltic states', 'Belarus, Ukraine, Moldova',
                'Norway', 'Sweden', 'Finland', 'Russia, European (< 60 E)', 'Russia, W Siberia (60-90 E)',
                'Russia, C Siberia (90-120 E)', 'Russia, E Siberia + Far East (> 120 E)', 'Alaska', 'Canada', 'Greenland']
MAPS = {'submitted': 'map_submitted', 'revised': 'map_revised'}
DIFFS = [('revised', 'submitted')]
KEYS = ['recall_peat', 'recall_nonpeat', 'BA', 'OA', 'precision', 'F1']
CNT = ['peat_as_peat', 'peat_as_nonpeat', 'nonpeat_as_nonpeat', 'nonpeat_as_peat']   # confusion counts (TP, FN, TN, FP)
MIN_N, NBOOT = 30, 1000
RES, BUF = 1 / 1200, 20                                      # 3 arc-second pixels; 20-pixel tile buffer


def metrics(C):
    tp, fn, tn, fp = np.moveaxis(np.asarray(C, float), -1, 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        rp, rn, pr = tp / (tp + fn), tn / (tn + fp), tp / (tp + fp)
        return {'recall_peat': rp, 'recall_nonpeat': rn, 'BA': (rp + rn) / 2, 'OA': (tp + tn) / (tp + fn + tn + fp),
                'precision': pr, 'F1': 2 * pr * rp / (pr + rp)}


def valid(n1, n0):
    return {'recall_peat': n1 >= MIN_N, 'recall_nonpeat': n0 >= MIN_N, 'BA': min(n1, n0) >= MIN_N, 'OA': n1 + n0 >= MIN_N,
            'precision': min(n1, n0) >= MIN_N, 'F1': min(n1, n0) >= MIN_N}


def eval_group(d, maps, head):
    y = d.label.values
    _, blk = np.unique(d.block.values, return_inverse=True)
    nb = int(blk.max()) + 1
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    ok = valid(n1, n0)
    R = np.random.default_rng(0).multinomial(nb, np.full(nb, 1 / nb), size=NBOOT).astype(float)   # block bootstrap weights
    info = {**head, 'n_peat': n1, 'n_nonpeat': n0, 'blocks_peat': int(d.block[y == 1].nunique()),
            'blocks_nonpeat': int(d.block[y == 0].nunique())}
    rows, drows = [], []
    nsame = d.groupby(['block', 'label']).label.transform('size').values
    for wname, w in (('points', np.ones(len(d))), ('blocks', 1.0 / nsame)):
        C = {}
        for m in maps:
            p = d[m].values
            cell = np.select([(y == 1) & (p == 1), (y == 1) & (p == 0), (y == 0) & (p == 0)], [0, 1, 2], 3)
            C[m] = np.bincount(blk * 4 + cell, weights=w, minlength=nb * 4).reshape(nb, 4)
        base = {m: metrics(C[m].sum(0)) for m in maps}
        boot = {m: metrics(R @ C[m]) for m in maps}
        for m in maps:
            r = {**info, 'weighting': wname, 'map': m}
            r.update(zip(CNT, C[m].sum(0).round(3)))
            for k in KEYS:
                r[k] = float(base[m][k]) if ok[k] else np.nan
            if ok['BA']:
                r['BA_lo'], r['BA_hi'] = np.nanpercentile(boot[m]['BA'], [2.5, 97.5])
            rows.append(r)
        for a, c in DIFFS:
            if a not in maps or c not in maps:
                continue
            dr = {**info, 'weighting': wname, 'map': a, 'vs': c}
            for k in KEYS:
                if ok[k]:
                    dr[f'd_{k}'] = float(base[a][k] - base[c][k])
                    dr[f'd_{k}_lo'], dr[f'd_{k}_hi'] = np.nanpercentile(boot[a][k] - boot[c][k], [2.5, 97.5])
            drows.append(dr)
    return rows, drows


def groups(D):
    out = [('All regions', np.ones(len(D), bool)), ('Eurasia', ~D.na), ('Europe (excl. Russia)', ~D.na & (D.country != 'Russia')),
           ('Russia', D.country == 'Russia'), ('North America', D.na), ('United Kingdom', D.region.str.startswith('UK: '))] + \
          [(g, D.region == g) for g in REGIONS]
    return out + [(f'dataset: {s}', D.dataset == s) for s in sorted(D.dataset.unique())]


def pixel_centres(pixel_id):
    """centre (lat, lon) of the pixels '<tile>_<row>_<col>' of the 5 x 5 deg tile grid (6040 x 6040 px with a 20 px buffer)"""
    p = pd.Series(pixel_id).str.split('_', expand=True)
    t, row, col = p[0], p[1].astype(int).values, p[2].astype(int).values
    lat0 = t.str[1:3].astype(int).values * np.where(t.str[0] == 'N', 1, -1)
    lon0 = t.str[4:7].astype(int).values * np.where(t.str[3] == 'E', 1, -1)
    lat = lat0 + 5 + (BUF - 1 - row) * RES
    lon = lon0 + (col - (BUF - 1)) * RES
    return lat, np.where(lon > 180, lon - 360, lon)


def read_map(path, pixel_id):
    """map value at the pixel centres (window per 5 deg tile)"""
    import rasterio
    from rasterio.windows import Window
    lat, lon = pixel_centres(pixel_id)
    out = np.full(len(lat), 255, 'uint8')
    with rasterio.open(path) as r:
        T = r.transform                                      # north-up grid: row and column of the cell holding the centre
        rows = np.floor((lat - T.f) / T.e).astype(np.int64)
        cols = np.floor((lon - T.c) / T.a).astype(np.int64)
        tile = pd.Series(pixel_id).str[:7].values
        for t in np.unique(tile):
            k = np.flatnonzero((tile == t) & (rows >= 0) & (rows < r.height) & (cols >= 0) & (cols < r.width))
            if not len(k):
                continue
            r0, c0 = rows[k].min(), cols[k].min()
            a = r.read(1, window=Window(c0, r0, cols[k].max() - c0 + 1, rows[k].max() - r0 + 1))
            out[k] = a[rows[k] - r0, cols[k] - c0]
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--points', default=str(POINTS))
    ap.add_argument('--map', help='GeoTIFF read instead of the column map_revised')
    ap.add_argument('--all-pixels', action='store_true', help='do not restrict to used_in_table3 = 1')
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()
    D = pd.read_csv(a.points)
    D = D.rename(columns={'block_05deg': 'block', 'north_america': 'na', **{v: k for k, v in MAPS.items()}})
    D['na'] = D.na.astype(bool)
    if a.map:
        stored = D.revised.values.copy()
        D['revised'] = read_map(a.map, D.pixel_id.values)
        print(f'{a.map}: read at {len(D):,} pixels; same value as map_revised at {(D.revised.values == stored).sum():,}',
              flush=True)
    maps = list(MAPS)
    on = np.logical_and.reduce([D[m].values != 255 for m in maps])
    if not a.all_pixels:
        on &= D.used_in_table3.values == 1
    rows, drows = [], []
    for name, mask in groups(D):
        mask = np.asarray(mask)
        d = D[mask & on]
        if len(d) < MIN_N:
            continue
        r, dr = eval_group(d, maps, {'group': name, 'n_not_used': int((mask & ~on).sum())})
        rows += r
        drows += dr
    R, Q = pd.DataFrame(rows), pd.DataFrame(drows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / 'evaluation_metrics.csv', index=False)
    Q.to_csv(out / 'evaluation_difference.csv', index=False)
    pd.set_option('display.width', 200)
    show = R[R.weighting == 'points'].pivot_table(index='group', columns='map', values='BA', sort=False)
    print('balanced accuracy (pixels):')
    print(show.round(3).to_string())
    print('->', out / 'evaluation_metrics.csv')
