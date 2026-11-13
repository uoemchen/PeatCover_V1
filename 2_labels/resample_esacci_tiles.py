"""Resample the ESA CCI Land Cover 2022 map (300 m) to the Landsat tiles (nearest neighbour).

The labels use class 210 (water bodies).
Input:  ESA_CCI_LC_2022.tif (GeoTIFF of C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1), Landsat 8 tiles
Output: Processed/2_PEATMAP/ESACCI/Europe/<tile>/<tile>_CCI_LC.tif (Europe)
        Processed/2_PEATMAP/ESACCI/USA/<tile>/<tile>_CCI_LC.tif (North America)
Usage:  python resample_esacci_tiles.py europe|usa
"""
import argparse
import os
import subprocess
from pathlib import Path

import rasterio
from rasterio.warp import Resampling, reproject

ROOT = Path(os.environ.get('PEATCOVER_ROOT', 'path/to/PEATLAND'))
CCI = ROOT / 'Raw/6_Others/ESA_CCI_2022/ESA_CCI_LC_2022.tif'
REGIONS = {'europe': (ROOT / 'Raw/1_Landsat8/Landsat8_Europe_July',
                      ROOT / 'Processed/2_PEATMAP/ESACCI/Europe'),
           'usa': (ROOT / 'Raw/1_Landsat8/Landsat8_NA_July',
                   ROOT / 'Processed/2_PEATMAP/ESACCI/USA')}


def resample_rasterio(ref, out_file):
    with rasterio.open(CCI) as src:
        profile = src.profile.copy()
        profile.update({'crs': ref.crs, 'transform': ref.transform, 'width': ref.width, 'height': ref.height,
                        'compress': 'lzw'})
        with rasterio.open(out_file, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                          src_transform=src.transform, src_crs=src.crs, dst_transform=ref.transform,
                          dst_crs=ref.crs, resampling=Resampling.nearest)


def resample_gdalwarp(ref, out_file):
    b = ref.bounds
    cmd = ['gdalwarp', '-of', 'GTiff', '-r', 'near', '-te', str(b.left), str(b.bottom), str(b.right), str(b.top),
           '-ts', str(ref.width), str(ref.height), '-t_srs', str(ref.crs), '-co', 'COMPRESS=LZW', '-co', 'TILED=YES',
           str(CCI), str(out_file)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main(region):
    landsat, out_dir = REGIONS[region]
    for tif in sorted(landsat.glob('*.tif')):
        tile_name = tif.stem.split('_')[-1]
        out = out_dir / tile_name
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / f'{tile_name}_CCI_LC.tif'
        if out_file.exists():
            continue
        with rasterio.open(tif) as ref:
            try:
                resample_rasterio(ref, out_file)
            except Exception as e:                         # fall back to gdalwarp
                print(f'{tile_name}: rasterio failed ({e}), using gdalwarp')
                resample_gdalwarp(ref, out_file)
        print('SAVED', out_file)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('region', choices=sorted(REGIONS))
    main(p.parse_args().region)
