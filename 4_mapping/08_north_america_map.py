"""North-American map (Canada + Alaska).

Per 5 deg tile, on the pixel lattice of the released map (pixel centres at k/1200 deg):
  part 1  PEATMAP Canada polygons                      polygon downscaling of 05 (vb_lakes.tif)
  part 2  region2 (USA incl. Alaska, 0 % polygons)      tile thresholds of 06 (cover_tiles)
  part 3  northern Canada                               PI >= 82 (06, cover_north)
  180-165 W strip                                       from the Eurasian 180 deg tiles (03)
HydroLAKES lakes are non-peat; the map is clipped to the published extent.
Steps (python 08_north_america_map.py <tiles|mosaic> [--workers 12]):
  tiles   -> WORK/na_final/{v17iv, v17, part}/<tile>.tif, results/tables/na_final_area*.csv
  mosaic  -> Prediction_Output/PeatCover/NorthAmerica_17_plus_OPERA_IV/*.tif (+ .tif.ovr)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse  # noqa: E402
import importlib  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import geopandas as gpd  # noqa: E402
from affine import Affine  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.windows import Window, from_bounds, transform as win_transform  # noqa: E402
from scipy.ndimage import binary_dilation, distance_transform_edt, label  # noqa: E402

from config import BUF, ROOT, TAB, WORK, tile_centres, tile_transform  # noqa: E402

eurasia = importlib.import_module('03_eurasia_map')   # write_tif
canada = importlib.import_module('05_canada_regression_vb')             # masks, points, metrics
na_pi = importlib.import_module('06_north_america_pi')                # tiles, lakes, cover tiles

PUB = canada.HALD / 'PeatCover_submitted_NorthAmerica.tif'
COUNTRIES = ROOT / 'Raw/Study_Area/Pan_boreal/Northern_study_area_country.shp'
TMP = WORK / 'na_final'
OUT = ROOT / 'Prediction_Output/PeatCover/NorthAmerica_17_plus_OPERA_IV'
VERSIONS = ('v17iv', 'v17')
PARTS = {1: 'PEATMAP Canada polygons (VB)', 2: 'region2 (tile thresholds)', 3: 'outside region2 (PI >= T)',
         4: '180-175 W (Eurasian tiles)', 5: 'seam between the masks (tile thresholds)',
         6: 'hole in the PEATMAP Canada polygons (PI >= T)'}
SEAM_R = 10                                                  # px, closing radius (gaps up to ~2 x SEAM_R px wide)
SEAM_M = 100                                                 # px read around the core for the closing and the pieces
REGIONS = ['Canada', 'Alaska', 'Russia', 'Greenland', '(no country)']
ZONES = {0: 'south of 75 N', 1: 'north of 75 N'}
COARSE = 60                                                  # 0.05 deg cells for the figure
KM2 = (6371.0088 * np.pi / 180 / 1200) ** 2
G = {}


def all_tiles():
    """NA tiles of step 06 plus the Eurasian E180 tiles (180-175 W), named by their W origin"""
    strip = sorted(f'N{p.stem[1:3]}W180' for p in (WORK / 'eurasia_cmp/v17iv/cover').glob('N*E180.tif'))
    return na_pi.tiles() + strip


def init():
    C = gpd.read_file(COUNTRIES)[['ADMIN', 'geometry']].to_crs(4326)
    G['countries'] = C[C.ADMIN.isin(REGIONS)]
    G['pub'] = rasterio.open(PUB)


def read_core(path, tr_core):
    with rasterio.open(path) as r:
        w = from_bounds(tr_core.c, tr_core.f - 5, tr_core.c + 5, tr_core.f, transform=r.transform).round_offsets().round_lengths()
        return r.read(1, window=w, boundless=True, fill_value=0)


def window_of(path, tr, fill, n=6000):
    with rasterio.open(path) as r:
        col = int(round((tr.c - r.transform.c) / r.transform.a))
        row = int(round((r.transform.f - tr.f) / -r.transform.e))
        return r.read(1, window=Window(col, row, n, n), boundless=True, fill_value=fill)


def seams(tr):
    """part-3 pixels between the PEATMAP Canada mask and region2: (seam, hole) in the tile core (see the module docstring)"""
    m = SEAM_M
    trm = tr * Affine.translation(-m, -m)
    n = 6000 + 2 * m
    can = window_of(canada.CAN_MASK, trm, 0, n) == 1
    r2 = window_of(canada.REGION2, trm, 0, n) == 1
    a = can | r2
    if a.all() or not a.any():
        z = np.zeros((6000, 6000), bool)
        return z, z
    gap = (distance_transform_edt(distance_transform_edt(~a) <= SEAM_R) > SEAM_R) & ~a
    lab, _ = label(gap, structure=np.ones((3, 3), bool))
    touch = np.unique(lab[gap & binary_dilation(r2 & ~can, structure=np.ones((3, 3), bool))])
    seam = np.isin(lab, touch[touch > 0])
    core = (slice(m, m + 6000), slice(m, m + 6000))
    return seam[core], (gap & ~seam)[core]


def compose(tile, v, tr, sh=None):
    """map (0 / 1 / 255) and part (1-6) of one version in a tile core; sh = seams(tr)"""
    if tile.endswith('W180'):
        e180 = tile.replace('W180', 'E180')
        m = rasterio.open(WORK / 'eurasia_cmp' / v / 'cover' / f'{e180}.tif').read(1)
        m[na_pi.lakes_full(e180)[na_pi.CORE, na_pi.CORE]] = 0
        return m, np.full(m.shape, 4, 'uint8')
    can = read_core(canada.CAN_MASK, tr) == 1
    r2 = read_core(canada.REGION2, tr) == 1
    vb = window_of(canada.OUT / v / 'vb_lakes.tif', tr, 0)
    cov = {}
    for k in ('cover_tiles', 'cover_north'):
        f = na_pi.TMP / v / k / f'{tile}.tif'
        cov[k] = rasterio.open(f).read(1) if f.exists() else np.full((6000, 6000), 255, 'uint8')
    seam, hole = sh if sh is not None else seams(tr)
    m = np.where(can, (vb > 0).astype('uint8'), np.where(r2 | seam, cov['cover_tiles'], cov['cover_north']))
    m[(m == 255) & (na_pi.lake_class(tile)[na_pi.CORE, na_pi.CORE] > 0)] = 0      # lake without inputs (e.g. Great Bear Lake): non-peat
    part = np.where(can, 1, np.where(r2, 2, np.where(seam, 5, np.where(hole, 6, 3)))).astype('uint8')
    return m, part


def tile_job(tile):
    if tile.endswith('W180'):
        tr = rasterio.open(WORK / 'eurasia_cmp/v17iv/cover' / f'{tile.replace("W180", "E180")}.tif').transform
    else:
        tr = win_transform(Window(BUF, BUF, 6000, 6000), tile_transform(tile))
    pub = window_of(PUB, tr, 255)
    if (pub == 255).all():
        return tile, None
    M = {}
    sh = None if tile.endswith('W180') else seams(tr)
    for v in VERSIONS:
        m, part = compose(tile, v, tr, sh)
        m[pub == 255] = 255                                                   # clip to the published extent
        M[v] = m
        eurasia.write_tif(TMP / v / f'{tile}.tif', m, tr, 255, 'uint8')
    eurasia.write_tif(TMP / 'part' / f'{tile}.tif', np.where(pub != 255, part, 0).astype('uint8'), tr, 0, 'uint8')
    M['pub'] = pub
    ok = (pub != 255) & (M['v17iv'] != 255)
    ch = np.full((6000, 6000), 255, 'uint8')
    ch[ok] = np.select([(pub == 1) & (M['v17iv'] == 1), (pub == 1) & (M['v17iv'] == 0), (pub == 0) & (M['v17iv'] == 1)],
                       [1, 2, 3], 0)[ok]
    eurasia.write_tif(TMP / 'change_pub_vs_v17iv' / f'{tile}.tif', ch, tr, 255, 'uint8')
    # areas (km2) by country x part x zone
    lat, _ = tile_centres(tile)
    latc = lat[na_pi.CORE]
    w = (KM2 * np.cos(np.radians(latc)))[:, None] * np.ones((1, 6000))
    sel = G['countries'].cx[tr.c - 0.1:tr.c + 5.1, tr.f - 5.1:tr.f + 0.1]
    reg = (rasterize([(g, REGIONS.index(a) + 1) for g, a in zip(sel.geometry, sel.ADMIN)], out_shape=(6000, 6000), transform=tr,
                     fill=len(REGIONS), dtype='uint8') if len(sel) else np.full((6000, 6000), len(REGIONS), 'uint8')) - 1
    zone = np.broadcast_to((latc >= 75)[:, None], (6000, 6000)).astype('uint8')
    key = (reg.astype(int) * 10 + part) * 2 + zone
    land = pub != 255
    st = {}
    for name, mask in [('land', land), ('peat_pub', pub == 1), ('peat_v17iv', M['v17iv'] == 1), ('peat_v17', M['v17'] == 1),
                       ('both_pub_v17iv', (pub == 1) & (M['v17iv'] == 1)),
                       ('gap_v17iv', land & (M['v17iv'] == 255)), ('gap_v17', land & (M['v17'] == 255))]:
        k = key[mask]
        st[name] = np.bincount(k, weights=w[mask], minlength=len(REGIONS) * 20)
    # 0.05 deg peat fractions (share of the land of the published map) for the figure
    co = {}
    for name, m in (('land', land), ('pub', pub == 1), ('v17iv', M['v17iv'] == 1), ('v17', M['v17'] == 1)):
        co[name] = m.reshape(6000 // COARSE, COARSE, 6000 // COARSE, COARSE).sum((1, 3)).astype('float32')
    rows = np.nonzero(land.any(1))[0]
    cols = np.nonzero(land.any(0))[0]
    bbox = (tr.c + cols.min() * tr.a, tr.f + (rows.max() + 1) * tr.e, tr.c + (cols.max() + 1) * tr.a, tr.f + rows.min() * tr.e)
    return tile, (st, co, bbox, (tr.c, tr.f))


def step_tiles(workers):
    t0 = time.time()
    T = all_tiles()
    S, CO, B = {}, {}, []
    with Pool(workers, initializer=init) as pool:
        for i, (t, out) in enumerate(pool.imap_unordered(tile_job, T), 1):
            if out is not None:
                st, co, bbox, org = out
                S[t] = st
                CO[t] = (org, co)
                B.append(bbox)
            if i % 20 == 0 or i == len(T):
                print(f'  tiles {i}/{len(T)} {(time.time() - t0) / 60:.1f} min', flush=True)
    rows = []
    for t, st in S.items():
        for k in range(len(REGIONS) * 20):
            if st['land'][k] > 0 or st['peat_v17iv'][k] > 0:
                z, rp = k % 2, k // 2
                rows.append({'tile': t, 'region': REGIONS[rp // 10], 'part': PARTS.get(rp % 10, rp % 10), 'zone': ZONES[z],
                             **{n: a[k] for n, a in st.items()}})
    D = pd.DataFrame(rows)
    D.to_csv(TAB / 'na_final_area_by_tile.csv', index=False)
    b = np.array(B)
    bbox = (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())
    np.save(TMP / 'bbox.npy', np.array(bbox))
    np.savez_compressed(TMP / 'coarse.npz', **{f'{t}|{n}': a for t, (org, co) in CO.items() for n, a in co.items()},
                        **{f'{t}|org': np.array(org) for t, (org, co) in CO.items()})
    summary(D)


def summary(D=None):
    D = pd.read_csv(TAB / 'na_final_area_by_tile.csv') if D is None else D
    cols = ['land', 'peat_pub', 'peat_v17iv', 'peat_v17', 'both_pub_v17iv', 'gap_v17iv', 'gap_v17']
    out = []
    for keys in (['region'], ['region', 'part'], ['region', 'zone']):
        g = D.groupby(keys)[cols].sum().reset_index()
        g.insert(0, 'grouping', ' x '.join(keys))
        out.append(g)
    tot = D[cols].sum().to_frame().T
    tot.insert(0, 'grouping', 'total')
    R = pd.concat(out + [tot], ignore_index=True)
    R = R[['grouping', 'region', 'part', 'zone'] + cols]
    for v in ('pub', 'v17iv', 'v17'):
        R[f'share_{v} (%)'] = 100 * R[f'peat_{v}'] / R.land
    R['Dice pub vs v17iv (%)'] = 100 * 2 * R.both_pub_v17iv / (R.peat_pub + R.peat_v17iv)
    for c in cols:
        R[c] = R[c] / 1e3                                                    # 1000 km2
    R = R.rename(columns={c: f'{c} (1000 km2)' for c in cols})
    R.round(3).to_csv(TAB / 'na_final_area.csv', index=False)
    pd.set_option('display.width', 250)
    print(R.round(2).to_string(index=False), flush=True)


# ------------------------------------------------------------------------------------------------ step mosaic
CMAP_PEAT = {0: (230, 230, 230, 255), 1: (0, 109, 44, 255)}
CMAP_CHANGE = {0: (255, 255, 255, 255), 1: (127, 127, 127, 255), 2: (230, 85, 13, 255), 3: (49, 130, 189, 255)}
JOBS = [('v17iv', 'NA_peat_17inputs_plus_OPERA_IV.tif', CMAP_PEAT), ('v17', 'NA_peat_17inputs.tif', CMAP_PEAT),
        ('change_pub_vs_v17iv', 'NA_change_published_vs_IV.tif', CMAP_CHANGE)]


def mosaic_one(job):
    """per-tile files -> VRT -> one GeoTIFF over the published data extent (DEFLATE, every block written) + colour table +
    external pyramids (nearest, 2-512); built in OUT/_tmp and moved in place"""
    folder, name, cmap = job
    stage = OUT / '_tmp'
    stage.mkdir(parents=True, exist_ok=True)
    src = TMP / folder
    vrt = src.parent / f'{folder}.vrt'
    subprocess.run(['gdalbuildvrt', '-q', '-srcnodata', '255', '-vrtnodata', '255', str(vrt)] + sorted(str(p) for p in src.glob('*.tif')),
                   check=True)
    w, s, e, n = np.load(TMP / 'bbox.npy')
    out = stage / name
    (stage / f'{name}.ovr').unlink(missing_ok=True)
    subprocess.run(['gdal_translate', '-q', '-projwin', *(repr(float(x)) for x in (w, n, e, s)), '-of', 'GTiff', '-co', 'COMPRESS=DEFLATE',
                    '-co', 'TILED=YES', '-co', 'BLOCKXSIZE=512', '-co', 'BLOCKYSIZE=512', '-co', 'BIGTIFF=YES', '-co', 'NUM_THREADS=4',
                    str(vrt), str(out)], check=True)
    with rasterio.open(out, 'r+') as dst:
        dst.write_colormap(1, cmap)
    subprocess.run(['gdaladdo', '-q', '-ro', '-r', 'nearest', '--config', 'COMPRESS_OVERVIEW', 'DEFLATE', '--config', 'BIGTIFF_OVERVIEW',
                    'YES', '--config', 'GDAL_NUM_THREADS', '4', str(out), '2', '4', '8', '16', '32', '64', '128', '256', '512'], check=True)
    os.replace(out, OUT / name)
    os.replace(stage / f'{name}.ovr', OUT / f'{name}.ovr')
    return name, (OUT / name).stat().st_size / 1e9, (OUT / f'{name}.ovr').stat().st_size / 1e9


def step_mosaic(workers):
    t0 = time.time()
    with Pool(min(workers, len(JOBS))) as pool:
        for name, gb, ogb in pool.imap_unordered(mosaic_one, JOBS):
            print(f'  {name}: {gb:.2f} GB + .ovr {ogb:.2f} GB, {(time.time() - t0) / 60:.1f} min', flush=True)
    (OUT / '_tmp').rmdir()
    with rasterio.open(OUT / JOBS[0][1]) as a, rasterio.open(PUB) as p:
        print('grid check: column offset to the published clip', (a.transform.c - p.transform.c) * 1200,
              'row offset', (p.transform.f - a.transform.f) * 1200, a.width, a.height)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['tiles', 'mosaic'])
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    if a.step == 'tiles':
        step_tiles(a.workers)
    if a.step == 'mosaic':
        step_mosaic(a.workers)
