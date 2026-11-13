"""Eurasia: Peatland Index (PI) of every tile and the downscaled map, with 17 inputs and with 18 inputs.

The same code for both input sets; PI = mean of the 3 classification networks (seeds 1-3), on the full 6040 x 6040 tile:
  v17      17 inputs                                        f17_h10_full_s1-3
  v17iv    18 inputs (17 + OPERA inundated vegetation)      f17iv_h10_full_s1-3
Downscaling with the functions of 02_thresholds.py (tile thresholds to the reference fraction, neighbour filling,
bilinear interpolation).
Merge of the overlapping tiles: PI = mean of the valid tiles; peat if any valid tile says peat (max PI >= threshold), so
that the buffers fill the thin gaps of a tile's own Landsat 8 data.
Steps:
  pi       per tile: the networks on the full tile -> PI (uint16, 1e-4) of each input set + core thresholds
  merge    per tile and input set: merged core -> PI 0-100 and peat map; change maps against the published map; areas
  mosaic   one GeoTIFF per map + external pyramids (.tif.ovr)
Outputs: WORK/eurasia_cmp/<version>/{pi_u16, pi_u8, cover}/, Prediction_Output/PeatCover/Eurasia_OPERA_comparison/
Usage: python 03_eurasia_map.py [--step pi merge mosaic] [--workers 12] [--tiles ...]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'     # single-threaded BLAS: the worker pools are forked
import argparse  # noqa: E402
import importlib  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import geopandas as gpd  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from affine import Affine  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from config import BUF, N, OPERA90, ROOT, WORK, input_cols, stack_path, tile_centres, tile_origin, tile_transform  # noqa: E402
from matlab_net import MatlabNet  # noqa: E402

thresholds = importlib.import_module('02_thresholds')
VERSIONS = {'v17': '17', 'v17iv': '17iv'}                  # version -> input set (config.INPUT_SETS)
N_OPERA = {'17': 0, '17iv': 1, '19': 2}                                 # OPERA columns after the 17 inputs: IV, then OW
NETS = {v: [f'f{s}_h10_full_s{k}' for k in (1, 2, 3)] for v, s in VERSIONS.items()}
FILE = {'pub': 'published', 'v17': '17inputs', 'v17iv': '17inputs_plus_OPERA_IV'}
SHORT = {'pub': 'published', 'v17': '17', 'v17iv': 'IV'}
CHANGES = [('pub', 'v17iv')]                                            # change maps (A, B)
PAIRS = [('pub', 'v17'), ('pub', 'v17iv'), ('v17', 'v17iv')]
ALL = ['pub'] + list(VERSIONS)
OUT = ROOT / 'Prediction_Output/PeatCover/Eurasia_OPERA_comparison'
TMP = WORK / 'eurasia_cmp'
CORE = slice(BUF, BUF + 6000)
CHUNK = 2_000_000
G = {}


def full_transform(tile):
    """transform of the full 6040 x 6040 tile. Tiles of the 180 deg strip that start at 180 E are shifted by -360 (the boundary
    polygons stop at 180): their 20 buffer columns west of 180 E then lie outside the polygons and stay empty; the E175 tiles
    cover them."""
    tr = tile_transform(tile)
    return Affine(tr.a, tr.b, tr.c - 360, tr.d, tr.e, tr.f) if tile_origin(tile)[1] >= 180 else tr


def land_full(tile):
    """inside the Eurasian boundary (Europe.shp, as for the tile thresholds), full tile; the core part is identical to
    step 02's land mask (same pixel centres)"""
    tr = full_transform(tile)
    sel = G['boundary'].cx[tr.c - 0.1:tr.c + 5.2, tr.f - 5.2:tr.f + 0.1]
    if len(sel) == 0:
        return np.zeros((N, N), bool)
    return rasterize([(g, 1) for g in sel.geometry], out_shape=(N, N), transform=tr, fill=0, dtype='uint8') == 1


def write_tif(path, arr, transform, nodata, dtype):
    prof = dict(driver='GTiff', width=arr.shape[1], height=arr.shape[0], count=1, dtype=dtype, nodata=nodata, crs='EPSG:4326',
                transform=transform, compress='deflate', tiled=True, blockxsize=512, blockysize=512)
    if dtype == 'uint16':
        prof['predictor'] = 2
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.tif')
    with rasterio.open(tmp, 'w', **prof) as dst:
        dst.write(arr, 1)
    tmp.rename(path)


def tile_threshold(u, target):
    """threshold (as a PI value) from the stored PI u (uint16, 1e-4) of the valid core pixels, as 4_calculate_optimal_thresholds.py
    and step 02: the k-th largest value, k = ceil(n * target / 100); target <= 0 -> 1.0, k >= n -> 0.0"""
    n = len(u)
    if n == 0:
        return np.nan, np.nan
    k = int(np.ceil(n * target / 100))
    if target <= 0:
        return 1.0, 100 * np.mean(u >= 10000)
    if target >= 100 or k >= n:
        return 0.0, 100.0
    uk = np.partition(u, n - k)[n - k]
    return uk / 10000, 100 * np.mean(u >= uk)


# ------------------------------------------------------------------------------------------------ step 1: PI of every version
def pi_tile(args):
    tile, target = args
    t0 = time.time()
    files = {v: TMP / v / 'pi_u16' / f'{tile}.tif' for v in VERSIONS}
    res = {'tile': tile, 'target': target}
    land = land_full(tile)
    if not land.any():
        res['note'] = 'no land'
        return res
    if all(f.exists() for f in files.values()):
        for v, f in files.items():
            with rasterio.open(f) as r:
                u = r.read(1)[CORE, CORE].ravel()
            u = u[u != 65535]
            res[f'{v}_threshold'], res[f'{v}_actual_pct'] = tile_threshold(u, target)
            res[f'{v}_valid_core'] = len(u)
        return res
    # the strip across 180 deg: the separate Dateline stacks (the Europe stacks of these tiles have no WTD)
    region = 'Dateline' if tile in G['strip'] else None
    with h5py.File(stack_path(tile, region), 'r') as h:
        raw = h['input'][:].reshape(17, N * N)            # pixel index = row + col * N (MATLAB column order)
    land_s = land.T.ravel()                               # the same pixel order as the stack
    idx = np.flatnonzero(land_s & np.isfinite(raw).all(0))
    X = raw[:, idx]                                       # (17, n)
    del raw
    with rasterio.open(OPERA90 / f'{tile}_opera.tif') as r:
        op = r.read().transpose(0, 2, 1).reshape(2, -1)[:, idx].T.astype('float32')     # (n, 2): IV, OW; stack order
    op[op == 255] = np.nan
    core_s = G['core'].ravel()[idx]                       # the core mask is symmetric: image order = stack order
    for v, s in VERSIONS.items():
        k = N_OPERA[s]
        assert input_cols(NETS[v][0]) == list(range(17)) + [17, 18][:k]
        good = np.isfinite(op[:, :k]).all(1)
        pi = np.empty(len(idx), 'float32')
        for s0 in range(0, len(idx), CHUNK):
            x = np.concatenate([X[:, s0:s0 + CHUNK].T, op[s0:s0 + CHUNK, :k]], 1) if k else X[:, s0:s0 + CHUNK].T
            pi[s0:s0 + CHUNK] = np.mean([G['nets'][m].predict(x) for m in NETS[v]], 0)
        u = np.full(N * N, 65535, 'uint16')
        u[idx[good]] = np.rint(pi[good] * 10000).astype('uint16')
        write_tif(files[v], u.reshape(N, N).T, full_transform(tile), 65535, 'uint16')
        uc = u[idx[good & core_s]]
        res[f'{v}_threshold'], res[f'{v}_actual_pct'] = tile_threshold(uc, target)
        res[f'{v}_valid_core'] = len(uc)
        res[f'{v}_valid_full'] = int(good.sum())
    res['minutes'] = (time.time() - t0) / 60
    return res


# ------------------------------------------------------------------------------------------------ step 2: merge + classify
def merged_pi(v, tile):
    """merged core of one version: sum and number of the valid PI values (uint16) of the tile and of its neighbours' buffers,
    and their maximum"""
    lat0, lon0u = thresholds.unwrapped_lon0(tile)
    tot = np.zeros((6000, 6000), 'uint32')
    cnt = np.zeros((6000, 6000), 'uint8')
    mx = np.zeros((6000, 6000), 'uint16')
    for a in (-1, 0, 1):                 # neighbour to the north (+1) / south (-1)
        for b in (-1, 0, 1):             # neighbour to the east (+1) / west (-1)
            nb = G['by_ll'].get((lat0 + 5 * a, lon0u + 5 * b))
            f = TMP / v / 'pi_u16' / f'{nb}.tif'
            if nb is None or not f.exists():
                continue
            # rows / cols of this tile's full grid covered by the neighbour: neighbour row = row + 6000 a, col = col - 6000 b
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


def merge_tile(tile):
    lat0, lon0u = thresholds.unwrapped_lon0(tile)
    lat, lon = tile_centres(tile)
    latc, lonc = lat[CORE], lon[CORE] + (lon0u - tile_origin(tile)[1])     # unwrapped longitudes for the threshold surface
    tr = thresholds.out_transform(tile)
    M = {}
    for v in VERSIONS:
        tot, cnt, mx = merged_pi(v, tile)
        have = cnt > 0
        if not have.any():
            M[v] = np.full((6000, 6000), 255, 'uint8')
            continue
        rr, cc = np.nonzero(have)
        thr = G['surface'][v](np.c_[latc[rr], lonc[cc]])
        ok = np.isfinite(thr)
        cover = np.full((6000, 6000), 255, 'uint8')
        cover[rr[ok], cc[ok]] = (mx[rr[ok], cc[ok]] / 10000.0 >= thr[ok]).astype('uint8')
        pi8 = np.full((6000, 6000), 255, 'uint8')
        pi8[have] = np.clip(np.rint(tot[have] / cnt[have] / 100.0), 0, 100).astype('uint8')
        write_tif(TMP / v / 'pi_u8' / f'{tile}.tif', pi8, tr, 255, 'uint8')
        write_tif(TMP / v / 'cover' / f'{tile}.tif', cover, tr, 255, 'uint8')
        M[v] = cover
    if all((M[v] == 255).all() for v in VERSIONS):
        return tile, None
    with rasterio.open(thresholds.PUBLISHED) as r:
        col = int(round((tr.c - r.transform.c) / tr.a))
        row = int(round((r.transform.f - tr.f) / -tr.e))
        M['pub'] = r.read(1, window=Window(col, row, 6000, 6000), boundless=True, fill_value=255)
    for a, b in CHANGES:
        ok = (M[a] != 255) & (M[b] != 255)
        ch = np.full((6000, 6000), 255, 'uint8')
        ch[ok] = np.select([(M[a] == 1) & (M[b] == 1), (M[a] == 1) & (M[b] == 0), (M[a] == 0) & (M[b] == 1)], [1, 2, 3], 0)[ok]
        write_tif(TMP / f'change_{SHORT[a]}_vs_{SHORT[b]}' / f'{tile}.tif', ch, tr, 255, 'uint8')
    # areas (km2) by country: each map on its own land, and all maps on the land where all four have data
    w = ((6371.0088 * np.pi / 180 / 1200) ** 2 * np.cos(np.radians(latc)))[:, None] * np.ones((1, 6000))
    w = w.ravel()
    sel = G['countries'].cx[tr.c - 0.1:tr.c + 5.1, tr.f - 5.1:tr.f + 0.1]
    code = (rasterize([(g, c + 1) for g, c in zip(sel.geometry, sel.code)], out_shape=(6000, 6000), transform=tr, fill=0,
                      dtype='int16').ravel() if len(sel) else np.zeros(6000 * 6000, 'int16'))
    nc = len(G['countries']) + 1
    flat = {k: m.ravel() for k, m in M.items()}
    common = np.logical_and.reduce([flat[k] != 255 for k in ALL])

    def bc(m):
        return np.bincount(code[m], weights=w[m], minlength=nc)
    st = {'land_common': bc(common)}
    for k in ALL:
        st[f'land_{k}'] = bc(flat[k] != 255)
        st[f'peat_{k}'] = bc(flat[k] == 1)
        st[f'peat_{k}_common'] = bc(common & (flat[k] == 1))
    for a, b in PAIRS:
        st[f'both_{a}_{b}_common'] = bc(common & (flat[a] == 1) & (flat[b] == 1))
    return tile, st


# ------------------------------------------------------------------------------------------------ step 3: mosaics
def mosaic_jobs():
    jobs = []
    for v in VERSIONS:
        jobs.append((TMP / v / 'pi_u8', f'Eurasia_PI_0to100_{FILE[v]}.tif', 'average', None))
        jobs.append((TMP / v / 'cover', f'Eurasia_peat_{FILE[v]}.tif', 'nearest', {0: (230, 230, 230, 255), 1: (0, 109, 44, 255)}))
    for a, b in CHANGES:
        jobs.append((TMP / f'change_{SHORT[a]}_vs_{SHORT[b]}', f'Eurasia_change_{SHORT[a]}_vs_{SHORT[b]}.tif', 'nearest',
                     {0: (255, 255, 255, 255), 1: (127, 127, 127, 255), 2: (230, 85, 13, 255), 3: (49, 130, 189, 255)}))
    return jobs


def build_vrt(src):
    """VRT of the per-tile files of one map (<folder>.vrt next to the folder; used by the mosaics)"""
    vrt = src.parent / f'{src.name}.vrt'
    files = sorted(str(p) for p in src.glob('*.tif'))
    subprocess.run(['gdalbuildvrt', '-q', '-srcnodata', '255', '-vrtnodata', '255', str(vrt)] + files, check=True)
    return vrt


def mosaic_one(job):
    """per-tile files -> VRT -> one GeoTIFF (DEFLATE, every block written: no SPARSE_OK, which ArcGIS may not read) + colour
    table + external pyramids (.tif.ovr, levels 2-512). Built in OUT/_tmp, then moved in place (a copy may be running)."""
    src, name, rs, cmap = job
    stage = OUT / '_tmp'
    vrt = build_vrt(src)
    out = stage / name
    (stage / f'{name}.ovr').unlink(missing_ok=True)
    subprocess.run(['gdal_translate', '-q', '-of', 'GTiff', '-co', 'COMPRESS=DEFLATE', '-co', 'TILED=YES', '-co', 'BLOCKXSIZE=512',
                    '-co', 'BLOCKYSIZE=512', '-co', 'BIGTIFF=YES', '-co', 'NUM_THREADS=4', str(vrt), str(out)], check=True)
    if cmap:
        with rasterio.open(out, 'r+') as dst:
            dst.write_colormap(1, cmap)
    subprocess.run(['gdaladdo', '-q', '-ro', '-r', rs, '--config', 'COMPRESS_OVERVIEW', 'DEFLATE', '--config', 'BIGTIFF_OVERVIEW',
                    'YES', '--config', 'GDAL_NUM_THREADS', '4', str(out), '2', '4', '8', '16', '32', '64', '128', '256', '512'],
                   check=True)
    (OUT / f'{name}.aux.xml').unlink(missing_ok=True)
    os.replace(out, OUT / name)
    os.replace(stage / f'{name}.ovr', OUT / f'{name}.ovr')
    return name, (OUT / name).stat().st_size / 1e9, (OUT / f'{name}.ovr').stat().st_size / 1e9


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', nargs='*', default=['pi', 'merge', 'mosaic'])
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--tiles', nargs='*', help='only these tiles (test)')
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    T = thresholds.tiles_and_targets()
    G['strip'] = {f.name.replace('_probability.tif', '') for f in thresholds.PI_STRIP.glob('*_probability.tif')}
    G['by_ll'] = {thresholds.unwrapped_lon0(t): t for t in T}
    G['core'] = np.zeros((N, N), bool)
    G['core'][CORE, CORE] = True
    if a.tiles:
        T = {t: v for t, v in T.items() if t in a.tiles}
    fthr = TMP / 'thresholds.csv'
    if 'pi' in a.step:
        G['boundary'] = gpd.read_file(thresholds.BOUNDARY).to_crs(4326)
        G['nets'] = {m: MatlabNet(WORK / 'nn' / f'{m}.mat') for ms in NETS.values() for m in ms}
        t0 = time.time()
        rows = []
        jobs = sorted(T.items())
        with Pool(a.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(pi_tile, jobs), 1):
                rows.append(r)
                if i % 10 == 0:
                    print(f'  PI {i}/{len(jobs)} tiles {(time.time() - t0) / 60:.1f} min', flush=True)
        D = pd.DataFrame(rows).sort_values('tile')
        if a.tiles and fthr.exists():                     # test run: keep the other tiles' rows
            old = pd.read_csv(fthr)
            D = pd.concat([old[~old.tile.isin(D.tile)], D]).sort_values('tile')
        D.to_csv(fthr, index=False)
        print('PI + thresholds done', len(D), f'{(time.time() - t0) / 60:.1f} min', flush=True)
    if 'merge' in a.step:
        D = pd.read_csv(fthr)
        G['surface'] = {}
        for v in VERSIONS:
            d = {thresholds.unwrapped_lon0(t): x for t, x in zip(D.tile, D[f'{v}_threshold'])}
            G['surface'][v] = thresholds.threshold_surface(thresholds.expand_origins(d))
        C = gpd.read_file(thresholds.COUNTRIES)[['ADMIN', 'geometry']].to_crs(4326)
        C['code'] = np.arange(len(C))
        G['countries'] = C
        t0 = time.time()
        tot = None
        tiles = sorted(T)
        with Pool(a.workers) as pool:
            for i, (t, st) in enumerate(pool.imap_unordered(merge_tile, tiles), 1):
                if st is not None:
                    tot = st if tot is None else {k: tot[k] + st[k] for k in tot}
                if i % 20 == 0:
                    print(f'  merge {i}/{len(tiles)} tiles {(time.time() - t0) / 60:.1f} min', flush=True)
        S = pd.DataFrame(tot, index=['(no country polygon)'] + list(C.ADMIN))
        S.index.name = 'country'
        S.to_csv(TMP / 'area_by_country_raw.csv')
        for src, _, _, _ in mosaic_jobs():
            build_vrt(src)
        print('merge done', f'{(time.time() - t0) / 60:.1f} min', flush=True)
    if 'mosaic' in a.step:
        (OUT / '_tmp').mkdir(exist_ok=True)
        with Pool(3) as pool:
            for name, gb, ovr in pool.imap_unordered(mosaic_one, mosaic_jobs()):
                print(f'wrote {name} {gb:.2f} GB + pyramids {ovr:.2f} GB', flush=True)
        (OUT / '_tmp').rmdir()
