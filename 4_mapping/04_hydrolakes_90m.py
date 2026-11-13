"""HydroLAKES v1.0 on the 90 m tile grid (North America).

Rasterises the HydroLAKES polygons (cell centre in polygon) on every 6040 x 6040 tile; value = lake size class:
0 no lake, 1 < 1 km2, 2 1-10, 3 10-100, 4 100-1000, 5 1000-10000, 6 >= 10000 km2 (larger lakes burnt last).
Lake pixels are not candidates of the Canadian polygon downscaling (05), are masked in the North-American tile thresholds
(06) and are non-peat in the North-American map (08).
Output: Processed/HydroLAKES_90m_tilegrid/<tile>_lakes.tif + a VRT mosaic
Usage: python 04_hydrolakes_90m.py [--workers 12]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse  # noqa: E402
import subprocess  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pyogrio  # noqa: E402
import rasterio  # noqa: E402
import shapely  # noqa: E402
from rasterio.features import rasterize  # noqa: E402

from config import N, OPERA90, ROOT, tile_transform  # noqa: E402

SHP = ROOT / 'Peat_Wetland_Dataset/3_Ancillary/HydroLAKES_v10/raw/extracted/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp'
OUT = ROOT / 'Processed/HydroLAKES_90m_tilegrid'
EDGES = [0, 1, 10, 100, 1000, 10000, np.inf]            # km2 -> classes 1..6
G = {}


def tiles():
    return sorted(p.name.split('_')[0] for p in OPERA90.glob('*_opera.tif'))


def tile_lakes(tile):
    f = OUT / f'{tile}_lakes.tif'
    if f.exists():
        return tile, 'exists'
    tr = tile_transform(tile)
    b = (tr.c, tr.f - N * tr.a, tr.c + N * tr.a, tr.f)
    idx = G['tree'].query(shapely.box(*b))
    geoms = G['geom'][idx]
    if b[2] > 180:                                       # E180 tiles of the 180 deg strip: lakes west of -175 shifted by 360
        j = G['tree'].query(shapely.box(b[0] - 360, b[1], b[2] - 360, b[3]))
        idx = np.r_[idx, j]
        geoms = np.r_[geoms, shapely.transform(G['geom'][j], lambda c: c + [360, 0])]
    a = np.zeros((N, N), 'uint8')
    if len(idx):
        o = np.argsort(G['area'][idx])                   # small first, large lakes burnt last
        a = rasterize(zip(geoms[o], G['cls'][idx][o]), out_shape=(N, N), transform=tr, fill=0, dtype='uint8')
    prof = dict(driver='GTiff', height=N, width=N, count=1, dtype='uint8', crs='EPSG:4326', transform=tr,
                compress='deflate', tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(f, 'w', **prof) as r:
        r.write(a, 1)
    return tile, int((a > 0).sum())


def build_tiles(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    m, _, geom, fld = pyogrio.raw.read(SHP, columns=['Lake_area'])
    G['area'] = fld[0].astype('float64')
    G['cls'] = np.digitize(G['area'], EDGES[1:-1]) + 1
    G['geom'] = shapely.from_wkb(geom)
    G['tree'] = shapely.STRtree(G['geom'])
    print('lakes read', len(G['geom']), flush=True)
    T = tiles()
    with Pool(workers) as pool:
        for k, (t, n) in enumerate(pool.imap_unordered(tile_lakes, T)):
            if k % 50 == 0:
                print(k, t, n, flush=True)
    subprocess.run(['gdalbuildvrt', str(OUT / 'HydroLAKES_90m_mosaic.vrt')] + [str(OUT / f'{t}_lakes.tif') for t in T],
                   check=True, capture_output=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    build_tiles(a.workers)
