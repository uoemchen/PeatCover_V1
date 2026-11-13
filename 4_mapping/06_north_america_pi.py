"""North America outside the Canadian PEATMAP polygons: classification PI and per-tile downscaling.

  classification PI   the networks of 03 (v17, v17iv) on the North-American stacks (full 6040 x 6040 tiles)
  region2             USA (incl. Alaska) and the Canadian polygons with 0 % peat: per-tile thresholds as in Eurasia, target =
                      peatfra_al of NA_reference_fraction.shp (1_inputs/reference_peat_fraction.py), threshold pixels
                      inside USAandothers.shp; neighbour filling, one ring, bilinear interpolation
  northern Canada     PI >= NORTH (82 for v17iv; 07_north_threshold.py)
Steps (python 06_north_america_pi.py <pi|classify> [--workers 10]):
  pi        PI (uint16, 1e-4) + tile thresholds -> WORK/na_cmp/<v>/pi_u16/
  classify  merged cores: PI 0-100, map with tile thresholds, map with the northern threshold
            -> WORK/na_cmp/<v>/{pi_u8, cover_tiles, cover_north}/<tile>.tif
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'     # single-threaded BLAS: the worker pools are forked
import argparse  # noqa: E402
import importlib  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402
import rasterio  # noqa: E402
import shapely  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.windows import Window, transform as win_transform  # noqa: E402

from config import BUF, N, OPERA90, PRED_DB, ROOT, TAB, WORK, tile_centres, tile_origin, tile_transform  # noqa: E402
from matlab_net import MatlabNet  # noqa: E402

thresholds = importlib.import_module('02_thresholds')        # expand_origins, threshold_surface (tile-centre keys)
eurasia = importlib.import_module('03_eurasia_map')   # tile_threshold, write_tif

A = ROOT / 'Analysis'
REF = A / 'reference_fraction/NorthAmerica/NA_reference_fraction.shp'
BOUNDARY = ROOT / 'Raw/Study_Area/USAandothers.shp'
LAKES = ROOT / 'Processed/HydroLAKES_90m_tilegrid'
LAKE_MIN = 1                                                 # smallest HydroLAKES size class masked (1 = all lakes)
TMP = WORK / 'na_cmp'
VERSIONS = {'v17': 0, 'v17iv': 1}                            # version -> number of OPERA columns (IV)
NETS = {v: [f'f{v[1:]}_h10_full_s{k}' for k in (1, 2, 3)] for v in VERSIONS}
NORTH = {'v17': 73, 'v17iv': 82}
CORE = slice(BUF, BUF + 6000)
CHUNK = 2_000_000
G = {}


def tiles():
    """W tiles of the NA stacks from 35 N (thresholds of the tiles around Canada and Alaska) and east of 175 W (the 175-180 W
    strip comes from the Eurasian 180 deg tiles, as in the published map)"""
    na = sorted(p.name.split('_')[0] for p in PRED_DB['NA'].glob('*_predictionData.mat'))
    return [t for t in na if t[3] == 'W' and int(t[1:3]) >= 35 and int(t[4:7]) <= 175]


def targets():
    m, _, _, f = pyogrio.raw.read(REF, read_geometry=False)
    D = pd.DataFrame(dict(zip(m['fields'], f)))
    return {t: (float(v) if h == 1 else None) for t, v, h in zip(D.tile_name, D.peatfra_al, D.has_data)}


def init():
    m, _, g, _ = pyogrio.raw.read(BOUNDARY, columns=[])
    G['boundary'] = shapely.from_wkb(g)
    G['nets'] = {n: MatlabNet(WORK / 'nn' / f'{n}.mat') for ns in NETS.values() for n in ns}


def boundary_full(tile):
    tr = tile_transform(tile)
    box = shapely.box(tr.c, tr.f - N / 1200, tr.c + N / 1200, tr.f)
    sel = [x for x in G['boundary'] if shapely.intersects(x, box)]
    if not sel:
        return np.zeros((N, N), bool)
    return rasterize([(x, 1) for x in sel], out_shape=(N, N), transform=tr, fill=0, dtype='uint8') == 1


def lake_class(tile):
    """HydroLAKES size class 0-6 of the full tile (step 04; 0 = no lake)"""
    f = LAKES / f'{tile}_lakes.tif'
    if not f.exists():
        return np.zeros((N, N), 'uint8')
    with rasterio.open(f) as r:
        return r.read(1)


def lakes_full(tile):
    """lakes that are masked: size class >= LAKE_MIN (1 = every HydroLAKES lake)"""
    return lake_class(tile) >= LAKE_MIN


# ------------------------------------------------------------------------------------------------ step pi
def pi_tile(args):
    tile, target = args
    t0 = time.time()
    files = {v: TMP / v / 'pi_u16' / f'{tile}.tif' for v in VERSIONS}
    bnd = boundary_full(tile)[CORE, CORE]
    lake = lakes_full(tile)
    res = {'tile': tile, 'target': target, 'usa_core_px': int(bnd.sum())}
    if not all(f.exists() for f in files.values()):
        with h5py.File(PRED_DB['NA'] / f'{tile}_predictionData.mat', 'r') as h:
            raw = h['input'][:].reshape(17, N * N)        # pixel index = row + col * N (MATLAB column order)
        idx = np.flatnonzero(np.isfinite(raw).all(0))
        X = raw[:, idx]
        del raw
        with rasterio.open(OPERA90 / f'{tile}_opera.tif') as r:
            iv = r.read(1).T.ravel()[idx].astype('float32')                  # stack order
        iv[iv == 255] = np.nan
        for v, k in VERSIONS.items():
            good = np.isfinite(iv) if k else np.ones(len(idx), bool)
            pi = np.empty(len(idx), 'float32')
            for s0 in range(0, len(idx), CHUNK):
                x = X[:, s0:s0 + CHUNK].T
                if k:
                    x = np.concatenate([x, iv[s0:s0 + CHUNK, None]], 1)
                pi[s0:s0 + CHUNK] = np.mean([G['nets'][m].predict(x) for m in NETS[v]], 0)
            u = np.full(N * N, 65535, 'uint16')
            u[idx[good]] = np.rint(pi[good] * 10000).astype('uint16')
            eurasia.write_tif(files[v], u.reshape(N, N).T, tile_transform(tile), 65535, 'uint16')
            res[f'{v}_iv_missing'] = int((~good).sum())
        res['minutes'] = (time.time() - t0) / 60
    nolake = ~lake[CORE, CORE]
    for v, f in files.items():
        with rasterio.open(f) as r:
            u = r.read(1)[CORE, CORE]
        ok = (u != 65535) & bnd & nolake
        res[f'{v}_threshold_px'] = int(ok.sum())
        if target is not None:
            res[f'{v}_threshold'], res[f'{v}_actual_pct'] = eurasia.tile_threshold(u[ok], target)
    return res


def step_pi(workers):
    t0 = time.time()
    T, tg = tiles(), targets()
    print(f'{len(T)} tiles, {sum(tg.get(t) is not None for t in T)} with a target', flush=True)
    rows = []
    with Pool(workers, initializer=init) as pool:
        for i, r in enumerate(pool.imap_unordered(pi_tile, [(t, tg.get(t)) for t in T]), 1):
            rows.append(r)
            if i % 10 == 0 or i == len(T):
                print(f'  {i}/{len(T)} tiles {(time.time() - t0) / 60:.1f} min', flush=True)
    R = pd.DataFrame(rows).sort_values('tile')
    R.to_csv(TAB / 'na_stage2_thresholds.csv', index=False)


# ------------------------------------------------------------------------------------------------ step classify
def surfaces():
    R = pd.read_csv(TAB / 'na_stage2_thresholds.csv')
    out = {}
    for v in VERSIONS:
        col = f'{v}_threshold'
        d = {tile_origin(t): (float(x) if pd.notna(x) else np.nan) for t, x in zip(R.tile, R[col])}
        out[v] = thresholds.expand_origins(d)
    return out


def init_classify():
    ex = surfaces()
    G['expanded'] = ex
    G['surface'] = {v: thresholds.threshold_surface(e) for v, e in ex.items()}
    G['by_ll'] = {tile_origin(t): t for t in tiles()}


def merged_pi(v, tile):
    """merged core: sum, number and maximum of the valid PI values of the tile and of its neighbours' buffers (step 03)"""
    lat0, lon0 = tile_origin(tile)
    tot = np.zeros((6000, 6000), 'uint32')
    cnt = np.zeros((6000, 6000), 'uint8')
    mx = np.zeros((6000, 6000), 'uint16')
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            nb = G['by_ll'].get((lat0 + 5 * a, lon0 + 5 * b))
            f = TMP / v / 'pi_u16' / f'{nb}.tif'
            if nb is None or not f.exists():
                continue
            r0, r1 = max(BUF, -6000 * a), min(BUF + 6000, N - 6000 * a)
            c0, c1 = max(BUF, 6000 * b), min(BUF + 6000, N + 6000 * b)
            with rasterio.open(f) as src:
                u = src.read(1, window=Window(c0 - 6000 * b, r0 + 6000 * a, c1 - c0, r1 - r0))
            ok = u != 65535
            sl = (slice(r0 - BUF, r1 - BUF), slice(c0 - BUF, c1 - BUF))
            tot[sl] += np.where(ok, u, 0)
            cnt[sl] += ok
            mx[sl] = np.maximum(mx[sl], np.where(ok, u, 0))
    return tot, cnt, mx


def classify_tile(tile):
    lat, lon = tile_centres(tile)
    latc, lonc = lat[CORE], lon[CORE]
    tr = win_transform(Window(BUF, BUF, 6000, 6000), tile_transform(tile))
    lake = lakes_full(tile)[CORE, CORE]
    out = {}
    for v in VERSIONS:
        tot, cnt, mx = merged_pi(v, tile)
        have = cnt > 0
        rr, cc = np.nonzero(have)
        thr = G['surface'][v](np.c_[latc[rr], lonc[cc]])
        ok = np.isfinite(thr)
        ct = np.full((6000, 6000), 255, 'uint8')
        ct[rr[ok], cc[ok]] = (mx[rr[ok], cc[ok]] / 10000.0 >= thr[ok]).astype('uint8')
        cn = np.full((6000, 6000), 255, 'uint8')
        cn[have] = (mx[have] / 10000.0 >= NORTH[v] / 100).astype('uint8')
        for c in (ct, cn):
            c[lake & (c != 255)] = 0                       # every HydroLAKES lake: non-peat
        pi8 = np.full((6000, 6000), 255, 'uint8')
        pi8[have] = np.clip(np.rint(tot[have] / cnt[have] / 100.0), 0, 100).astype('uint8')
        for name, arr in (('pi_u8', pi8), ('cover_tiles', ct), ('cover_north', cn)):
            eurasia.write_tif(TMP / v / name / f'{tile}.tif', arr, tr, 255, 'uint8')
        out[v] = int(have.sum())
    return tile, out


def step_classify(workers):
    t0 = time.time()
    init_classify()
    E = pd.DataFrame({v: pd.Series({f'N{k[0]:02d}W{-k[1]:03d}' if k[1] < 0 else f'N{k[0]:02d}E{k[1]:03d}': x for k, x in e.items()})
                      for v, e in G['expanded'].items()})
    E.index.name = 'tile'
    E.to_csv(TAB / 'na_stage2_thresholds_expanded.csv')
    T = tiles()
    with Pool(workers, initializer=init_classify) as pool:
        for i, (t, _) in enumerate(pool.imap_unordered(classify_tile, T), 1):
            if i % 20 == 0 or i == len(T):
                print(f'  classify {i}/{len(T)} {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['pi', 'classify'])
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    {'pi': step_pi, 'classify': step_classify}[a.step](a.workers)
