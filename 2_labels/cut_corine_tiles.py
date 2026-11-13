"""Cut the CORINE Land Cover 2018 polygons to the 5 x 5 deg Landsat tiles of Europe.

Input:  CORINE Land Cover 2018 vector (field Code_18), reprojected to EPSG:4326
        (Processed/2_PEATMAP/CORINE/CORINE_projected.gpkg), and the Landsat 8 tiles of Europe
Output: Processed/2_PEATMAP/CORINE/Tile_Shapefiles/<tile>/<tile>_CORINE.gpkg
Next:   rasterize_corine_tiles.py
"""
import os
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box

ROOT = Path(os.environ.get('PEATCOVER_ROOT', 'path/to/PEATLAND'))
CORINE = ROOT / 'Processed/2_PEATMAP/CORINE/CORINE_projected.gpkg'
LANDSAT = ROOT / 'Raw/1_Landsat8/Landsat8_Europe_July'
OUT = ROOT / 'Processed/2_PEATMAP/CORINE/Tile_Shapefiles'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corine = gpd.read_file(CORINE)
    corine['Code_18'] = corine['Code_18'].astype(str)
    print('Loaded CORINE:', len(corine), 'polygons')

    for tif in sorted(LANDSAT.glob('*.tif')):
        with rasterio.open(tif) as src:
            b = src.bounds
        tile_box = box(b.left, b.bottom, b.right, b.top)
        tile = corine[corine.intersects(tile_box)].copy()
        if tile.empty:
            continue

        # repair the geometries before and after clipping
        tile['geometry'] = tile['geometry'].buffer(0)
        tile = tile.clip(tile_box)
        invalid = ~tile.is_valid
        if invalid.sum() > 0:
            tile.loc[invalid, 'geometry'] = tile.loc[invalid, 'geometry'].buffer(0)

        tile_name = tif.stem.split('_')[-1]                  # e.g. N60E025
        out = OUT / tile_name
        out.mkdir(exist_ok=True)
        tile.to_file(out / f'{tile_name}_CORINE.gpkg', driver='GPKG')
        print(f'{tile_name}: {len(tile)} polygons')


if __name__ == '__main__':
    main()
