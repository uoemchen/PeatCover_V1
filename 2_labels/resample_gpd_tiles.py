"""Resample the Global Peatland Database map (GPA 2022, 2 classes, 1 km) to the Landsat tiles (Europe and North America).

Mode resampling; output values: 0 = no peat / no data, 1 and 2 = the two GPD peat classes (the labels use class 1 only).
Input:  GPD raster peatGPA22WGS_2cl.tif (EPSG:4326), Landsat 8 tiles
Output: Processed/2_PEATMAP/GPD/Tile_Rasters/<tile>/<tile>_GPD.tif (int8)
"""
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

ROOT = Path(os.environ.get('PEATCOVER_ROOT', 'path/to/PEATLAND'))
GPD = ROOT / 'Raw/2_PEATMAP/GPD/peatGPA22WGS_2cl.tif'
LANDSAT = [ROOT / 'Raw/1_Landsat8/Landsat8_Europe_July', ROOT / 'Raw/1_Landsat8/Landsat8_NA_July']
OUT = ROOT / 'Processed/2_PEATMAP/GPD/Tile_Rasters'


def resample_tile(gpd_data, gpd_src, ref, out_file):
    shape = (ref.height, ref.width)
    temp = np.empty(shape, dtype=gpd_src.dtypes[0])
    reproject(source=gpd_data, destination=temp, src_transform=gpd_src.transform, src_crs=gpd_src.crs,
              src_nodata=gpd_src.nodata, dst_transform=ref.transform, dst_crs=ref.crs, dst_nodata=0,
              resampling=Resampling.mode)
    out = np.zeros(shape, dtype=np.uint8)
    out[temp > 0] = temp[temp > 0]
    if gpd_src.nodata is not None:
        out[temp == gpd_src.nodata] = 0
    profile = {'driver': 'GTiff', 'dtype': 'int8', 'width': ref.width, 'height': ref.height, 'count': 1,
               'crs': ref.crs, 'transform': ref.transform, 'compress': 'lzw', 'nodata': None}
    with rasterio.open(out_file, 'w', **profile) as dst:
        dst.write(out, 1)


def main():
    tiles = {}                                             # tile name -> Landsat file (first region wins)
    for folder in LANDSAT:
        for tif in sorted(folder.glob('*.tif')):
            tiles.setdefault(tif.stem.split('_')[-1], tif)

    with rasterio.open(GPD) as gpd_src:
        gpd_data = gpd_src.read(1)
        for tile_name, tif in tiles.items():
            out = OUT / tile_name
            out.mkdir(parents=True, exist_ok=True)
            out_file = out / f'{tile_name}_GPD.tif'
            if out_file.exists():
                continue
            with rasterio.open(tif) as ref:
                resample_tile(gpd_data, gpd_src, ref, out_file)
            print('SAVED', out_file)


if __name__ == '__main__':
    main()
