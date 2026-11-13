"""Cut the PEATMAP polygons to the 5 x 5 deg Landsat tiles (Europe and the United States).

Input:  PEATMAP polygons in EPSG:4326 and the Landsat 8 tiles (GeoTIFF) that define the tile grid
Output: <out>/<tile>/<tile>_PEATMAP.shp
Next:   Peatmap_Rasterize.m
Usage:  python cut_peatmap_tiles.py europe|usa
"""
import argparse
import os
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box

ROOT = Path(os.environ.get('PEATCOVER_ROOT', 'path/to/PEATLAND'))
REGIONS = {
    'europe': dict(peatmap=ROOT / 'Raw/2_PEATMAP/Europe/Peatmap_European_Continent.shp',
                   landsat=ROOT / 'Raw/1_Landsat8/Landsat8_Europe_July',
                   out=ROOT / 'Processed/2_PEATMAP/Europe/Tile_Shapefiles'),
    'usa': dict(peatmap=ROOT / 'Raw/2_PEATMAP/NA/USA_and_others.shp',
                landsat=ROOT / 'Raw/1_Landsat8/Landsat8_NA_July',
                out=ROOT / 'Processed/2_PEATMAP/USA/Tile_Shapefiles')}


def main(region):
    cfg = REGIONS[region]
    cfg['out'].mkdir(parents=True, exist_ok=True)
    peatmap = gpd.read_file(cfg['peatmap'])
    print('Loaded PEATMAP:', len(peatmap), 'polygons')

    for tif in sorted(cfg['landsat'].glob('*.tif')):
        with rasterio.open(tif) as src:
            b = src.bounds
        tile_box = box(b.left, b.bottom, b.right, b.top)
        tile = peatmap[peatmap.intersects(tile_box)].copy()
        if tile.empty:
            continue
        tile = tile.clip(tile_box)

        tile_name = tif.stem.split('_')[-1]                  # e.g. N60E025
        out = cfg['out'] / tile_name
        out.mkdir(exist_ok=True)
        tile.to_file(out / f'{tile_name}_PEATMAP.shp')
        print(f'{tile_name}: {len(tile)} polygons')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('region', choices=sorted(REGIONS))
    main(p.parse_args().region)
