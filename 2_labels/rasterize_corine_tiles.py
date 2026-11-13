"""Rasterise the CORINE tile polygons (cut_corine_tiles.py) on the Landsat tile grid.

A pixel takes the CORINE code (Code_18, e.g. 412 = peat bogs) of the polygon that contains its centre; 0 = no polygon.
Output: Processed/2_PEATMAP/CORINE/Tile_Rasters/<tile>/<tile>_CORINE.tif (uint16)
"""
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

ROOT = Path(os.environ.get('PEATCOVER_ROOT', 'path/to/PEATLAND'))
TILES = ROOT / 'Processed/2_PEATMAP/CORINE/Tile_Shapefiles'
LANDSAT = ROOT / 'Raw/1_Landsat8/Landsat8_Europe_July'
OUT = ROOT / 'Processed/2_PEATMAP/CORINE/Tile_Rasters'


def write(out_file, data, ref):
    profile = {'driver': 'GTiff', 'dtype': 'uint16', 'nodata': 0, 'width': ref.width, 'height': ref.height, 'count': 1,
               'crs': ref.crs, 'transform': ref.transform, 'compress': 'lzw', 'tiled': True}
    with rasterio.open(out_file, 'w', **profile) as dst:
        dst.write(data, 1)


def rasterize_tile(corine_file, landsat_file, out_file):
    with rasterio.open(landsat_file) as ref:
        shape = (ref.height, ref.width)
        corine = gpd.read_file(corine_file)
        if str(corine.crs) != str(ref.crs):
            corine = corine.to_crs(ref.crs)
        corine['Code_18'] = pd.to_numeric(corine['Code_18'])
        corine = corine[corine['Code_18'].notna() & (corine['Code_18'] > 0)]
        shapes = [(g, int(c)) for g, c in zip(corine.geometry, corine['Code_18']) if g.is_valid and not g.is_empty]
        if shapes:
            data = rasterize(shapes, out_shape=shape, transform=ref.transform, fill=0, dtype=np.uint16,
                             all_touched=False)
        else:
            data = np.zeros(shape, dtype=np.uint16)
        write(out_file, data, ref)


def main():
    for tile_dir in sorted(d for d in TILES.iterdir() if d.is_dir()):
        tile_name = tile_dir.name
        corine_files = list(tile_dir.glob('*_CORINE.gpkg'))
        landsat_files = list(LANDSAT.glob(f'*{tile_name}.tif'))
        if not corine_files or not landsat_files:
            continue
        out = OUT / tile_name
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / f'{tile_name}_CORINE.tif'
        if out_file.exists():
            continue
        rasterize_tile(corine_files[0], landsat_files[0], out_file)
        print('SAVED', out_file)


if __name__ == '__main__':
    main()
