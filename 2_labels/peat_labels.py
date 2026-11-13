"""Peat / non-peat labels on the Landsat tiles of Europe and the United States (90 m).

Rules (1 = peat, 0 = non-peat, -1 = not used):
  - outside the study area (Europe and USA boundaries): -1
  - inside the CORINE coverage (where CORINE has a code): peat where PEATMAP, GPD (class 1) and CORINE (412, peat bogs)
    all say peat; -1 where only one or two of them say peat; 0 elsewhere
  - outside the CORINE coverage: peat where PEATMAP and GPD (class 1) both say peat; -1 where only one says peat;
    0 elsewhere
  - peat pixels that are water in ESA CCI 2022 (class 210) -> 0
Inputs: the tile rasters of Peatmap_Rasterize.m, resample_gpd_tiles.py, rasterize_corine_tiles.py and
        resample_esacci_tiles.py; the boundaries below.
Output: Training_database/label_tiles/<tile>_labels.tif (int8) and summary.json
Next:   BuildTrainingData.m
"""
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

ROOT = Path(os.environ.get('PEATCOVER_ROOT', 'path/to/PEATLAND'))
PEATMAP = [ROOT / 'Processed/2_PEATMAP/Europe/Tile_Rasters',
           ROOT / 'Processed/2_PEATMAP/USA/Tile_Rasters']
GPD = ROOT / 'Processed/2_PEATMAP/GPD/Tile_Rasters'
CORINE = ROOT / 'Processed/2_PEATMAP/CORINE/Tile_Rasters'
CCI = [ROOT / 'Processed/2_PEATMAP/ESACCI/Europe', ROOT / 'Processed/2_PEATMAP/ESACCI/USA']
LANDSAT = [ROOT / 'Raw/1_Landsat8/Landsat8_Europe_July', ROOT / 'Raw/1_Landsat8/Landsat8_NA_July']
OUT = ROOT / 'Training_database/label_tiles'

EUROPE_BOUNDARY = ROOT / 'Raw/2_PEATMAP/Europe/Boundary/Europe.shp'
NORTH_AMERICA_BOUNDARY = ROOT / 'Raw/Study_Area/USAandothers.shp'
CORINE_COVERAGE = ROOT / 'Raw/Study_Area/CORINE_coverage.shp'

CORINE_PEAT_CLASS = 412
WATER_CLASS = 210


def first(folders, pattern):
    for folder in folders:
        files = list(Path(folder).glob(pattern))
        if files:
            return files[0]
    return None


def read(path, shape=None, default=0):
    """int16 array of the first band (masked values -> default); a constant array when the file is missing"""
    if path is None or not path.exists():
        return None if shape is None else np.full(shape, default, dtype=np.int16)
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(default)
    return data.astype(np.int16)


def burn(gdf, transform, shape):
    return rasterize([(g, 1) for g in gdf.geometry], out_shape=shape, transform=transform, fill=0, dtype=np.uint8)


def labels_for_tile(tile, boundaries):
    landsat = first(LANDSAT, f'*{tile}.tif')
    peatmap_file = first(PEATMAP, f'{tile}/*PEAT*.tif')
    gpd_file = first([GPD], f'{tile}/*GPD*.tif')
    if landsat is None or peatmap_file is None or gpd_file is None:
        return None

    with rasterio.open(landsat) as src:
        profile, transform, shape = src.profile, src.transform, (src.height, src.width)
    peatmap = read(peatmap_file)
    gpd_ = read(gpd_file)
    corine = read(first([CORINE], f'{tile}/*CORINE*.tif'), shape, 0)
    cci = read(first(CCI, f'{tile}/*CCI*.tif'), shape, 0)
    if peatmap.shape != shape or gpd_.shape != shape:
        return None

    study = np.maximum(burn(boundaries['europe'], transform, shape), burn(boundaries['north_america'], transform, shape))
    corine_cov = burn(boundaries['corine_coverage'], transform, shape)

    labels = np.full(shape, -1, dtype=np.int8)
    labels[study > 0] = 0

    # CORINE coverage: three sources
    in_corine = (corine_cov > 0) & (study > 0) & (corine > 0)
    peat3 = (peatmap == 1) & (gpd_ == 1) & (corine == CORINE_PEAT_CLASS) & in_corine
    labels[peat3] = 1
    any3 = ((peatmap == 1) | (gpd_ == 1) | (corine == CORINE_PEAT_CLASS)) & in_corine
    labels[any3 & ~peat3] = -1

    # outside the CORINE coverage: two sources
    out_corine = (corine_cov == 0) & (study > 0)
    labels[(peatmap == 1) & (gpd_ == 1) & out_corine] = 1
    labels[((peatmap == 1) ^ (gpd_ == 1)) & out_corine] = -1

    # peat pixels that are water in ESA CCI -> non-peat
    labels[(cci == WATER_CLASS) & (labels == 1) & (study > 0)] = 0

    out_profile = profile.copy()
    out_profile.update({'dtype': 'int8', 'nodata': -1, 'compress': 'lzw', 'count': 1})
    with rasterio.open(OUT / f'{tile}_labels.tif', 'w', **out_profile) as dst:
        dst.write(labels, 1)

    stats = {'tile_name': tile, 'peat_pixels': int(np.sum(labels == 1)), 'nonpeat_pixels': int(np.sum(labels == 0)),
             'excluded_pixels': int(np.sum(labels == -1)), 'total_pixels': int(labels.size)}
    print(tile, stats)
    return stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    boundaries = {'europe': gpd.read_file(EUROPE_BOUNDARY), 'north_america': gpd.read_file(NORTH_AMERICA_BOUNDARY),
                  'corine_coverage': gpd.read_file(CORINE_COVERAGE)}
    tiles = sorted({d.name for folder in PEATMAP if folder.exists() for d in folder.iterdir()
                    if d.is_dir() and not d.name.startswith('.') and list(d.glob('*PEAT*.tif'))})
    print(len(tiles), 'tiles')

    stats = [s for s in (labels_for_tile(t, boundaries) for t in tiles) if s is not None]
    total = {k: sum(s[k] for s in stats) for k in ('peat_pixels', 'nonpeat_pixels', 'excluded_pixels')}
    with open(OUT / 'summary.json', 'w') as f:
        json.dump({'statistics': total, 'tiles': stats}, f, indent=2)
    print(total)


if __name__ == '__main__':
    main()
