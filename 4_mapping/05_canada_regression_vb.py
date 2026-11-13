"""Canada inside the PEATMAP polygons: regression network and polygon downscaling (VB).

Training data (2_labels/BuildTrainingData_Canada.m): every 90 m pixel inside a PEATMAP Canada polygon with a
water-corrected peat fraction of 1-100 % that is not MERIT permanent water; input = the 17 inputs (+ OPERA IV);
target = the peat fraction of the polygon (%). 56 of the 81 tiles are training tiles (data/networks/canada_tile_split.txt);
10 equal-width target bins, each cut to the size of the smallest one.
Network (3_training/run_can.m): patternnet(10), trainlm, mse, 50 epochs, 3 seeds; PI = mean of the 3 networks.
VB: in each polygon, the n pixels with the highest PI are peat, n = peat fraction x land pixels of the polygon; lake pixels
(HydroLAKES, 04) are not candidates, so the peat area of the polygon is kept on its land.
Steps (python 05_canada_regression_vb.py <step> [--version v17|v17iv] [--workers 12]):
  samples   balanced training sample -> WORK/canada/can_samples.mat (X: 17 inputs + IV, y, split); then 3_training/run_can.m
  predict   PI of every pixel of the Canadian tiles -> WORK/canada/<version>/pi.tif
  vb / vblakes   polygon downscaling -> WORK/canada/<version>/vb.tif, vb_lakes.tif (used by 08)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'     # single-threaded BLAS: the worker pools are forked
import argparse  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import scipy.io as sio  # noqa: E402

from rasterio.features import geometry_mask  # noqa: E402
from rasterio.windows import Window, from_bounds  # noqa: E402

from config import CANADA_SPLIT, OPERA90, ROOT, WORK, tile_centres  # noqa: E402
import pyogrio  # noqa: E402
import shapely  # noqa: E402

TRAIN_DB = ROOT / 'Training_database/canada_training_data'
SPLIT = CANADA_SPLIT
OUT = WORK / 'canada'
OUT.mkdir(exist_ok=True)
PO = ROOT / 'Prediction_Output'
PUB_PI = PO / 'PeatCover_submitted/Canada_PI.tif'          # uint8, water / no polygon = 0
SHP = PO / 'PeatCover_submitted/Canada_PEATMAP_polygons.shp'        # 5 820 polygon pieces, PEAT_PER = PEAT_PER_corr
HALD = PO / 'PeatCover_submitted'
CAN_MASK = HALD / 'Canada_PEATMAP_mask.tif'                         # 1 = inside a PEATMAP Canada polygon
REGION2 = HALD / 'NA_region2_mask.tif'                   # 1 = USA + PEATMAP Canada + 0 % ecozones
LAKES = ROOT / 'Processed/HydroLAKES_90m_tilegrid/HydroLAKES_90m_mosaic.vrt'    # lake size class 0-6 (0 = no lake), step 04
LAKE_MIN = 1                                  # smallest size class that is not a VB candidate
# versions: networks whose PI is averaged (as the Eurasian maps of step 03: mean of the seeds)
VERSIONS = {'v17': 'can17_s', 'v17iv': 'can17iv_s', **{f'v17_s{k}': f'can17_s{k}' for k in (1, 2, 3)},
            **{f'v17iv_s{k}': f'can17iv_s{k}' for k in (1, 2, 3)}}
EDGES = np.linspace(0, 100, 11)
TRAIN_SHARE, VAL_SHARE = 0.05, 0.01          # dividerand of nnModel2.m


def tiles():
    names = sorted(f.name.split('_')[0] for f in TRAIN_DB.glob('*_trainingData.mat'))
    txt = open(SPLIT).read().split('Test Files:')
    train = {s.split('_')[0] for s in txt[0].split() if s.endswith('.mat')}
    return names, train


def tbin(t):
    """bin index 0-9 as MATLAB discretize(t, 0:10:100) (last bin includes 100)"""
    return np.clip(np.searchsorted(EDGES, t, side='right') - 1, 0, 9)


def rowcol(tile, lat, lon):
    la, lo = tile_centres(tile)
    return np.rint((la[0] - lat) * 1200).astype(int), np.rint((lon - lo[0]) * 1200).astype(int)


def iv_at(tile, lat, lon):
    r, c = rowcol(tile, lat, lon)
    with rasterio.open(OPERA90 / f'{tile}_opera.tif') as s:
        a = s.read(1)
    v = a[r, c].astype('float32')
    v[v == 255] = np.nan
    return v


def count_bins(tile):
    with h5py.File(TRAIN_DB / f'{tile}_trainingData.mat') as h:
        t = h['target'][0]
    return tile, np.bincount(tbin(t), minlength=10)


def draw(args):
    tile, take, seed = args
    rng = np.random.default_rng(seed)
    with h5py.File(TRAIN_DB / f'{tile}_trainingData.mat') as h:
        t = h['target'][0]
        b = tbin(t)
        idx = np.sort(np.concatenate([rng.choice(np.where(b == k)[0], take[k], replace=False) for k in range(10) if take[k]]))
        X = h['input'][()][:, idx].T                     # h5py reads MATLAB N x 17 as 17 x N (whole array: point selection is slow)
        lat, lon = h['latS'][0][idx], h['lonS'][0][idx]
    return tile, np.hstack([X, iv_at(tile, lat, lon)[:, None]]), t[idx]


def samples(workers):
    t0 = time.time()
    names, train = tiles()
    tr = sorted(train)
    print(f'{len(names)} tiles, {len(tr)} training tiles', flush=True)
    with Pool(workers) as pool:
        C = dict(pool.map(count_bins, tr))
    M = np.array([C[t] for t in tr])                     # tiles x bins
    per_bin = M.sum(0)
    S = per_bin.min()
    m = int(round((TRAIN_SHARE + VAL_SHARE) * S))
    print('pixels', per_bin.sum(), 'per bin', per_bin.tolist(), 'smallest', S, '-> balanced', 10 * S, 'drawn per bin', m,
          f'{time.time() - t0:.0f}s', flush=True)
    rng = np.random.default_rng(42)
    take = np.stack([rng.multivariate_hypergeometric(M[:, k], m) for k in range(10)], 1)
    with Pool(workers) as pool:
        parts = pool.map(draw, [(t, take[i], 1000 + i) for i, t in enumerate(tr)])
    X = np.concatenate([p[1] for p in parts]).astype('float32')
    y = np.concatenate([p[2] for p in parts]).astype('float32')
    tile = np.concatenate([np.full(len(p[2]), i, 'int16') for i, p in enumerate(parts)])
    split = np.zeros(len(y), 'int8')
    for k in range(10):                                  # 1/6 of every bin to validation
        i = np.where(tbin(y) == k)[0]
        split[rng.choice(i, int(round(len(i) * VAL_SHARE / (TRAIN_SHARE + VAL_SHARE))), replace=False)] = 1
    print('n', len(y), 'train', int((split == 0).sum()), 'val', int((split == 1).sum()), 'IV missing', int(np.isnan(X[:, 17]).sum()),
          'any input missing', int((~np.isfinite(X)).any(1).sum()), 'y range', float(y.min()), float(y.max()),
          'per bin', np.bincount(tbin(y), minlength=10).tolist(), f'{time.time() - t0:.0f}s', flush=True)
    sio.savemat(OUT / 'can_samples.mat', {'X': X, 'y': y, 'split': split, 'tile': tile,
                                          'tiles': np.array(tr, dtype=object)}, do_compression=False)
    pd.DataFrame({'tile': tr, **{f'bin{k}': M[:, k] for k in range(10)}, **{f'take{k}': take[:, k] for k in range(10)}}).to_csv(
        OUT / 'can_samples_counts.csv', index=False)
    print('saved', OUT / 'can_samples.mat', f'{time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------------------------------------------------ PI maps
def nets_of(version):
    pre = VERSIONS[version]
    return sorted(f.stem for f in (OUT / 'nn').glob(f'{pre}*.mat'))


def predict_tile(args):
    """PI of every version on one tile; the inputs are read once and every network is applied once"""
    from matlab_net import MatlabNet
    tile, vnets = args
    with h5py.File(TRAIN_DB / f'{tile}_trainingData.mat') as h:
        X = h['input'][()].T
        lat, lon = h['latS'][0], h['lonS'][0]
    iv = iv_at(tile, lat, lon)[:, None]
    P = {}
    for n in sorted({n for ns in vnets.values() for n in ns}):
        P[n] = MatlabNet(OUT / f'nn/{n}.mat').predict(np.hstack([X, iv]) if 'iv' in n else X)
    r, c = rowcol(tile, lat, lon)
    out = {}
    for v, ns in vnets.items():
        p = np.mean([P[n] for n in ns], 0)
        a = np.zeros((6040, 6040), 'uint8')
        a[r, c] = np.where(np.isfinite(p), np.clip(p, 0, 100), 0).astype('uint8')     # truncated, as Mosaic_PI.py
        out[v] = a
    return tile, out, len(r), {v: int((~np.isfinite(P[ns[0]])).sum()) for v, ns in vnets.items()}


def predict(workers, versions):
    t0 = time.time()
    vnets = {v: nets_of(v) for v in versions.split(',')}
    names, _ = tiles()
    print('networks', vnets, flush=True)
    with rasterio.open(PUB_PI) as src:
        prof, tr = src.profile, src.transform
    M = {v: np.zeros((prof['height'], prof['width']), 'uint8') for v in vnets}
    n = 0
    with Pool(workers) as pool:
        for tile, out, k, bad in pool.imap_unordered(predict_tile, [(t, vnets) for t in names]):
            la, lo = tile_centres(tile)
            r0 = int(round((tr.f - (la[0] + 0.5 / 1200)) * 1200))
            c0 = int(round(((lo[0] - 0.5 / 1200) - tr.c) * 1200))
            for v, a in out.items():
                sub = M[v][r0:r0 + 6040, c0:c0 + 6040]   # the published mosaic stops 1 row short of the southern buffer
                a = a[:sub.shape[0], :sub.shape[1]]
                sub[a > 0] = a[a > 0]                     # buffers overlap the neighbour cores: same pixels, same inputs
            n += k
            print(f'{tile} {k:,} px, without a PI {bad}, {time.time() - t0:.0f}s', flush=True)
    prof.update(compress='deflate', tiled=True, blockxsize=512, blockysize=512, BIGTIFF='YES', nodata=0)
    for v, a in M.items():
        (OUT / v).mkdir(exist_ok=True)
        with rasterio.open(OUT / v / 'pi.tif', 'w', **prof) as dst:
            dst.write(a, 1)
        print(f'{v}: {n:,} px predicted, {int((a > 0).sum()):,} px with a PI, {time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------------------------------------------------ VB downscaling
def vb_polys(args):
    """VB of polygon pieces: 2 = peat in both rules, 1 = peat only with the published n (water counted twice).
    lakes=True: fixed rule only, lake pixels are not candidates (n from all land pixels) -> 1 = peat"""
    pi_file, items, lakes = args
    out = []
    with rasterio.open(pi_file) as src, rasterio.open(LAKES) as lk:
        T = src.transform
        for i, geom, pp in items:
            w = from_bounds(*geom.bounds, transform=T).round_offsets().round_lengths()
            c0, r0 = max(w.col_off - 1, 0), max(w.row_off - 1, 0)
            w = Window(c0, r0, min(w.width + 2, src.width - c0), min(w.height + 2, src.height - r0))
            a = src.read(1, window=w)
            m = geometry_mask([geom], out_shape=a.shape, transform=src.window_transform(w), invert=True)
            vals = a[m]
            valid = vals > 0
            if lakes:                                     # same lattice: offset of the PI raster in the lake mosaic
                lc = int(round((T.c - lk.transform.c) / T.a)) + w.col_off
                lr = int(round((lk.transform.f - T.f) / -T.e)) + w.row_off
                lake = lk.read(1, window=Window(lc, lr, w.width, w.height), boundless=True, fill_value=0)[m] >= LAKE_MIN
                n_land = int(valid.sum())
                cand = valid & ~lake
                n = min(int(round(float(pp) / 100 * n_land)), int(cand.sum()))
                res = None
                if n > 0:
                    order = np.argsort(vals[cand])[::-1]
                    sel = np.where(cand)[0][order[:n]]
                    rr, cc = np.where(m)
                    res = np.zeros(a.shape, 'uint8')
                    res[rr[sel], cc[sel]] = 1
                out.append((i, len(vals), n_land, int(cand.sum()), n, (r0, c0, res)))
                continue
            n_all, n_land = len(vals), int(valid.sum())
            n_orig = min(int(round(float(pp) / 100 * n_all)), n_land)              # downscaling_NA_test.py
            n_fixed = min(int(round(float(pp) / 100 * n_land)), n_land)
            res = None
            if n_orig > 0:
                order = np.argsort(vals[valid])[::-1]                               # same ranking (and ties) as the original downscaling
                sel = np.where(valid)[0][order[:n_orig]]
                rr, cc = np.where(m)
                res = np.zeros(a.shape, 'uint8')
                res[rr[sel], cc[sel]] = 1
                res[rr[sel[:n_fixed]], cc[sel[:n_fixed]]] = 2
            out.append((i, n_all, n_land, n_orig, n_fixed, (r0, c0, res)))
    return out


def vb(workers, version, lakes=False):
    t0 = time.time()
    pi_file = OUT / version / 'pi.tif'
    meta, _, geom, f = pyogrio.raw.read(SHP)
    F = dict(zip(meta['fields'], f))
    g = shapely.from_wkb(geom)
    order = np.argsort(-shapely.area(g))                  # largest first: balanced workers
    jobs = [(pi_file, [(int(i), g[i], F['PEAT_PER'][i]) for i in order[k::workers * 8]], lakes) for k in range(workers * 8)]
    with rasterio.open(pi_file) as src:
        prof, H, W, T = src.profile, src.height, src.width, src.transform
    V = np.zeros((H, W), 'uint8')
    rows = []
    with Pool(workers) as pool:
        for out in pool.imap_unordered(vb_polys, jobs):
            for i, n_all, n_land, n_orig, n_fixed, (r0, c0, res) in out:
                rows.append((i, F['PEAT_PER'][i], n_all, n_land, n_orig, n_fixed))
                if res is not None:
                    sub = V[r0:r0 + res.shape[0], c0:c0 + res.shape[1]]
                    np.maximum(sub, res, out=sub)
    lat = T.f + (np.arange(H) + 0.5) * T.e
    px = (6371.0088 * np.pi / 180 / 1200) ** 2 * np.cos(np.radians(lat))         # km2 per pixel of each row
    area = {k: float(((V >= k).sum(1) * px).sum()) for k in (1, 2)}
    (OUT / version).mkdir(exist_ok=True)
    prof.update(compress='deflate', tiled=True, blockxsize=512, blockysize=512, BIGTIFF='YES', nodata=None, dtype='uint8',
                NUM_THREADS=4)
    with rasterio.open(OUT / version / ('vb_lakes.tif' if lakes else 'vb.tif'), 'w', **prof) as dst:
        dst.write(V, 1)
    cols = ['poly', 'PEAT_PER', 'n_all', 'n_land', 'n_candidates', 'n_fixed'] if lakes else \
        ['poly', 'PEAT_PER', 'n_all', 'n_land', 'n_orig', 'n_fixed']
    P = pd.DataFrame(rows, columns=cols).sort_values('poly')
    P.to_csv(OUT / version / ('vb_lakes_polygons.csv' if lakes else 'vb_polygons.csv'), index=False)
    if lakes:
        print(f'{version}: VB fixed rule without lake pixels {area[1] / 1e3:.1f} k km2, {time.time() - t0:.0f}s', flush=True)
        return area
    print(f'{version}: VB peat area, published rule {area[1] / 1e3:.1f} k km2, fixed rule {area[2] / 1e3:.1f} k km2, '
          f'{time.time() - t0:.0f}s', flush=True)
    return area


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--version', default='v17')
    a = ap.parse_args()
    if a.step == 'predict':
        predict(a.workers, a.version)
    elif a.step in ('vb', 'vblakes'):
        for v in a.version.split(','):
            vb(a.workers, v, lakes=a.step == 'vblakes')
    else:
        {'samples': samples}[a.step](a.workers)
