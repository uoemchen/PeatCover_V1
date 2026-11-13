"""Reference peat fraction of each 5 x 5 deg tile from the a priori maps (Europe and the United States).

For each tile, on the 6000 x 6000 core pixels (the 20-pixel buffer of the Landsat tiles removed) and on land only:
  inside the CORINE coverage:   (PEATMAP + GPD + CORINE peat bogs) / 3 per pixel
  outside the CORINE coverage:  (PEATMAP + GPD) / 2 per pixel
(PEATMAP = 1 on a PEATMAP polygon; GPD = class 1 or 2; CORINE = code 412; a missing map counts as 0.)
peat_pct = sum over the land pixels / number of land pixels x 100.
Inputs: the tile rasters of 2_labels (PEATMAP, GPD, CORINE), the land boundary of the region, the CORINE coverage.
Output: Analysis/reference_fraction/<dir>/<REGION>_tile_fraction.shp and <REGION>_tile_fraction.csv
        (columns: tile_name, core_land, core_peat, peat_pct, peat_p / peat_g / peat_c = peat pixels of each map, ...)
These per-tile fractions were then combined in a GIS into <REGION>_reference_fraction.shp (field peatfra_al), read by
4_mapping/02_thresholds.py and 06_north_america_pi.py; the final values are in data/thresholds (reference_peat_fraction_pct).
Usage:  python reference_peat_fraction.py Europe|NA
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import box

ROOT = Path(os.environ.get('PEATCOVER_ROOT', 'path/to/PEATLAND'))
REGIONS = {
    'Europe': dict(landsat=ROOT / 'Raw/1_Landsat8/Landsat8_Europe_July',
                   peatmap=ROOT / 'Processed/2_PEATMAP/Europe/Tile_Rasters',
                   boundary=ROOT / 'Raw/2_PEATMAP/Europe/Boundary/Europe.shp',
                   out=ROOT / 'Analysis/reference_fraction/Europe'),
    'NA': dict(landsat=ROOT / 'Raw/1_Landsat8/Landsat8_NA_July',
               peatmap=ROOT / 'Processed/2_PEATMAP/USA/Tile_Rasters',
               boundary=ROOT / 'Raw/Study_Area/USAandothers.shp',
               out=ROOT / 'Analysis/reference_fraction/NorthAmerica')}
GPD_DIR = ROOT / 'Processed/2_PEATMAP/GPD/Tile_Rasters'
CORINE_DIR = ROOT / 'Processed/2_PEATMAP/CORINE/Tile_Rasters'
CORINE_COVERAGE = ROOT / 'Raw/Study_Area/CORINE_coverage.shp'

BUFFER_PIXELS = 20
CORE_SIZE = 6000
CORINE_PEAT_CLASS = 412
NUM_WORKERS = 3

_cfg = {}


def init_worker(region):
    _cfg['peatmap'] = REGIONS[region]['peatmap']
    _cfg['boundary'] = gpd.read_file(REGIONS[region]['boundary'])
    _cfg['corine_coverage'] = gpd.read_file(CORINE_COVERAGE)


def tile_files(tile):
    files = {'peatmap': _cfg['peatmap'] / tile / f'{tile}_PEATMASK.tif',
             'gpd': GPD_DIR / tile / f'{tile}_GPD.tif',
             'corine': CORINE_DIR / tile / f'{tile}_CORINE.tif'}
    return {k: (v if v.exists() else None) for k, v in files.items()}


def core_masks(files):
    """land mask and CORINE coverage mask on the core pixels of the tile grid"""
    ref = next(f for f in (files['peatmap'], files['gpd'], files['corine']) if f is not None)
    with rasterio.open(ref) as src:
        crs = src.crs
        core_tr = window_transform(Window(BUFFER_PIXELS, BUFFER_PIXELS, CORE_SIZE, CORE_SIZE), src.transform)
    masks = []
    for gdf in (_cfg['boundary'], _cfg['corine_coverage']):
        gdf = gdf.to_crs(crs) if gdf.crs != crs else gdf
        masks.append(rasterize([(g, 1) for g in gdf.geometry], out_shape=(CORE_SIZE, CORE_SIZE),
                               transform=core_tr, fill=0, dtype=np.uint8))
    return masks


def read_core(path, peat):
    """1 where the map says peat on the core pixels, or None when the raster is missing / too small"""
    if path is None:
        return None
    with rasterio.open(path) as src:
        full = src.read(1)
    if full.shape[0] < CORE_SIZE + 2 * BUFFER_PIXELS or full.shape[1] < CORE_SIZE + 2 * BUFFER_PIXELS:
        return None
    core = full[BUFFER_PIXELS:BUFFER_PIXELS + CORE_SIZE, BUFFER_PIXELS:BUFFER_PIXELS + CORE_SIZE]
    return peat(core).astype(np.float32)


def tile_statistics(tile):
    empty = {'tile_name': tile, 'core_pix': CORE_SIZE * CORE_SIZE, 'core_land': 0, 'core_peat': 0.0, 'peat_pct': 0.0,
             'peat_p': 0, 'peat_g': 0, 'peat_c': 0, 'corine_land': 0, 'has_data': False, 'data_src': 'None'}
    files = tile_files(tile)
    if all(v is None for v in files.values()):
        return empty
    land, corine_cov = core_masks(files)

    maps = {'P': read_core(files['peatmap'], lambda a: a == 1),
            'G': read_core(files['gpd'], lambda a: (a == 1) | (a == 2)),
            'C': read_core(files['corine'], lambda a: a == CORINE_PEAT_CLASS)}
    peat_pixels = {k: int(np.sum((v == 1) & (land == 1))) if v is not None else 0 for k, v in maps.items()}
    zero = np.zeros((CORE_SIZE, CORE_SIZE), dtype=np.float32)
    pm, gp, co = (maps[k] if maps[k] is not None else zero for k in 'PGC')

    in_corine = (corine_cov == 1) & (land == 1)
    out_corine = (corine_cov == 0) & (land == 1)
    avg = np.zeros((CORE_SIZE, CORE_SIZE), dtype=np.float32)
    avg[in_corine] = (pm[in_corine] + gp[in_corine] + co[in_corine]) / 3.0
    avg[out_corine] = (pm[out_corine] + gp[out_corine]) / 2.0

    n_land = int(np.sum(land == 1))
    stats = dict(empty, peat_p=peat_pixels['P'], peat_g=peat_pixels['G'], peat_c=peat_pixels['C'],
                 corine_land=int(np.sum(in_corine)))
    if n_land == 0:
        return stats
    core_peat = float(np.sum(avg[land == 1]))
    stats.update(core_land=n_land, core_peat=round(core_peat, 2), peat_pct=round(core_peat / n_land * 100, 2),
                 has_data=True, data_src='+'.join(k for k in 'PGC' if maps[k] is not None))
    print(tile, stats['peat_pct'], flush=True)
    return stats


def tile_box(tile):
    lat = int(tile[1:3]) * (1 if tile[0] == 'N' else -1)
    lon = int(tile[4:7]) * (1 if tile[3] == 'E' else -1)
    return box(lon, lat, lon + 5, lat + 5)


def main(region):
    cfg = REGIONS[region]
    cfg['out'].mkdir(parents=True, exist_ok=True)
    tiles = sorted({f.stem.split('_')[-1] for f in cfg['landsat'].glob('*.tif')}
                   | {d.name for d in cfg['peatmap'].iterdir() if d.is_dir()})
    with Pool(NUM_WORKERS, initializer=init_worker, initargs=(region,)) as pool:
        stats = pd.DataFrame(pool.map(tile_statistics, tiles))

    stats.to_csv(cfg['out'] / f'{region}_tile_fraction.csv', index=False)
    grid = gpd.GeoDataFrame(stats, geometry=[tile_box(t) for t in stats.tile_name], crs='EPSG:4326')
    grid.to_file(cfg['out'] / f'{region}_tile_fraction.shp')
    print(f'{int(stats.has_data.sum())} tiles with data ->', cfg['out'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('region', choices=sorted(REGIONS))
    main(p.parse_args().region)
