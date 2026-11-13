"""The 18th input: OPERA DSWx-S1 occurrence (Isik et al., 2026; 30 m) -> 90 m on the tile grid.

The OPERA layers (inundated vegetation and open water: % of the 2025 Sentinel-1 acquisitions) are averaged to the
3 arc-second cells of the other inputs (area-weighted mean, GDAL 'average'), i.e. the share of time a cell was inundated.
The inundated-vegetation layer (IV) is the 18th input of the networks.
Output: OPERA90/<tile>_opera.tif (band 1 IV, band 2 open water, uint8 %, 255 = no data) + a VRT mosaic.
Usage: python opera_90m.py [--workers 6] [--tiles N60E025 ...]
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from affine import Affine
from rasterio.warp import reproject

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '4_mapping'))     # config.py
from config import N, OPERA90, OPERA_NAMES, OPERA_SRC, all_tiles, tile_transform


def build(tile):
    out = OPERA90 / f'{tile}_opera.tif'
    if out.exists():
        return tile, 'skip', 0
    t0 = time.time()
    dst_tr = tile_transform(tile)
    # tiles east of 180 deg (E180: 180-185 E) lie at -180..-175 in the OPERA files: reproject on the wrapped grid
    src_tr_grid = Affine(dst_tr.a, dst_tr.b, dst_tr.c - 360, dst_tr.d, dst_tr.e, dst_tr.f) if dst_tr.c >= 179 else dst_tr
    bands = []
    for name in OPERA_NAMES:
        with rasterio.open(OPERA_SRC[name]) as src:
            a = np.full((N, N), np.nan, dtype='float32')
            reproject(rasterio.band(src, 1), a, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=src_tr_grid, dst_crs='EPSG:4326', dst_nodata=np.nan,
                      resampling=Resampling.average, num_threads=2)
        bands.append(a)
    prof = dict(driver='GTiff', width=N, height=N, count=2, dtype='uint8', crs='EPSG:4326', transform=dst_tr,
                nodata=255, compress='deflate', tiled=True, blockxsize=512, blockysize=512)
    tmp = out.with_suffix('.tmp.tif')
    with rasterio.open(tmp, 'w', **prof) as dst:
        for b, a in enumerate(bands, 1):
            dst.write(np.where(np.isfinite(a), np.clip(np.rint(a), 0, 100), 255).astype('uint8'), b)
        dst.set_band_description(1, 'inundated vegetation occurrence 2025 (%)')
        dst.set_band_description(2, 'open water occurrence 2025 (%)')
    tmp.rename(out)
    return tile, 'ok', time.time() - t0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--tiles', nargs='*')
    a = ap.parse_args()
    tiles = a.tiles or list(all_tiles())
    t0 = time.time()
    with Pool(a.workers) as p:
        for k, (t, s, dt) in enumerate(p.imap_unordered(build, tiles), 1):
            print(f'{k}/{len(tiles)} {t} {s} {dt:.0f}s  elapsed {(time.time() - t0) / 60:.1f} min', flush=True)
    # mosaic for GIS viewing (the tiles overlap by their 20 px buffers; the VRT simply shows the last one)
    import subprocess
    files = sorted(str(f) for f in OPERA90.glob('N*_opera.tif')) + sorted(str(f) for f in OPERA90.glob('S*_opera.tif'))
    (OPERA90 / 'filelist.txt').write_text('\n'.join(files))
    subprocess.run(['gdalbuildvrt', '-input_file_list', str(OPERA90 / 'filelist.txt'), str(OPERA90 / 'OPERA_90m_mosaic.vrt')],
                   check=False)
    print('== OPERA 90 m done', flush=True)
