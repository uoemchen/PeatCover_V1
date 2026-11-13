"""Downscaling functions: reference peat fractions of the tiles and the threshold surface (used by 03 and 06).

Downscaling of the classification PI:
  1. per 5 deg tile, the threshold that makes the peat share of the 6000 x 6000 core pixels inside the land boundary equal
     to the reference peat fraction of the tile (Europe_reference_fraction.shp, Dateline_reference_fraction.shp for the
     tiles across 180 deg; per-tile fractions from 1_inputs/reference_peat_fraction.py);
  2. thresholds of 0 / missing -> mean of the valid neighbours, plus one ring of tiles;
  3. bilinear interpolation between the tile centres; peat = PI >= interpolated threshold.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'     # single-threaded BLAS: the worker pools are forked

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from affine import Affine  # noqa: E402
from rasterio.windows import Window, transform as win_transform  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from config import BUF, ROOT, tile_origin, tile_transform  # noqa: E402

A = ROOT / 'Analysis/reference_fraction'
REF_MAIN = A / 'Europe/Europe_reference_fraction.shp'
REF_STRIP = A / 'Dateline/Dateline_reference_fraction.shp'
PI_MAIN = ROOT / 'Prediction_Output/PeatCover_submitted/PI_Eurasia_tiles'
PI_STRIP = ROOT / 'Prediction_Output/PeatCover_submitted/PI_Dateline_tiles'
BOUNDARY = ROOT / 'Raw/2_PEATMAP/Europe/Boundary/Europe.shp'
COUNTRIES = ROOT / 'Raw/Study_Area/Pan_boreal/Northern_study_area_country.shp'
PUBLISHED = ROOT / 'Prediction_Output/PeatCover_submitted/PeatCover_submitted.tif'


def unwrapped_lon0(tile):
    """tile origin longitude on a continuous axis: tiles west of 100 W are the strip beyond 180 (-> +360)"""
    lat0, lon0 = tile_origin(tile)
    return lat0, lon0 + 360 if lon0 < -100 else lon0


def out_transform(tile):
    """transform of the 6000 x 6000 core, longitudes wrapped to -180..180 (the published lattice)"""
    tr = win_transform(Window(BUF, BUF, 6000, 6000), tile_transform(tile))
    return tr if tr.c < 180 else Affine(tr.a, tr.b, tr.c - 360, tr.d, tr.e, tr.f)


def tiles_and_targets():
    """{tile: target %} (None = PI only, no own threshold) for the main Eurasian tiles and the 180 deg strip"""
    out = {}
    main = {f.name.replace('_probability.tif', '') for f in PI_MAIN.glob('*_probability.tif')}
    strip = {f.name.replace('_probability.tif', '') for f in PI_STRIP.glob('*_probability.tif')}
    g = gpd.read_file(REF_MAIN)
    tg = {t: v for t, v, h in zip(g.tile_name, g.peatfra_al, g.has_data) if h == 1}
    for t in main:
        out[t] = tg.get(t)
    g = gpd.read_file(REF_STRIP)
    tg = {t: v for t, v, h in zip(g.tile_name, g.peatfra_al, g.has_data) if h == 1}
    for t in strip:
        out[t] = tg.get(t)
    return out


# ------------------------------------------------------------------------------------------------ step 2: classify
def expand_origins(d):
    """4b on {(lat0, lon0): threshold}: invalid (0 / missing) -> mean of valid 8-neighbours (else 1.0), plus one ring"""
    valid = {k: v for k, v in d.items() if np.isfinite(v) and v not in (0.0, -999.0)}
    out = dict(valid)
    for k, v in d.items():
        if k in valid:
            continue
        nb = [valid[(k[0] + a, k[1] + b)] for a in (-5, 0, 5) for b in (-5, 0, 5) if (a or b) and (k[0] + a, k[1] + b) in valid]
        out[k] = float(np.mean(nb)) if nb else 1.0
    ring = {}
    for k in list(out):
        for a in (-5, 0, 5):
            for b in (-5, 0, 5):
                q = (k[0] + a, k[1] + b)
                if q in out or q in ring:
                    continue
                nb = [out[(q[0] + c, q[1] + e)] for c in (-5, 0, 5) for e in (-5, 0, 5)
                      if (c or e) and (q[0] + c, q[1] + e) in out]
                ring[q] = float(np.mean(nb))
    out.update(ring)
    return out


def threshold_surface(expanded):
    la = np.array(sorted({k[0] + 2.5 for k in expanded}))
    lo = np.array(sorted({k[1] + 2.5 for k in expanded}))
    V = np.full((len(la), len(lo)), np.nan)
    ia, io = {v: i for i, v in enumerate(la)}, {v: i for i, v in enumerate(lo)}
    for k, v in expanded.items():
        V[ia[k[0] + 2.5], io[k[1] + 2.5]] = v
    return RegularGridInterpolator((la, lo), V, method='linear', bounds_error=False, fill_value=np.nan)


