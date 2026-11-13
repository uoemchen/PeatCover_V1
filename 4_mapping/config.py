"""Paths and grid definitions shared by the Python scripts (4_mapping/, 5_evaluation/, 1_inputs/opera_90m.py).

Set the environment variable PEATCOVER_ROOT to the data root (all input paths below are relative to it) and, optionally,
PEATCOVER_WORK to the folder of the large intermediate files.
Grid: 5 x 5 deg tiles of 6040 x 6040 pixels (6000 x 6000 core + 20 px buffer), 3 arc-second (1/1200 deg) pixels whose
centres lie on multiples of 1/1200 deg; MATLAB stacks are stored column-major (index = row + col * 6040).
"""
import os
from pathlib import Path

import numpy as np
import pyproj

# optional PROJ data directory (some conda environments ship proj.db where pyproj does not look by default)
if os.environ.get('PEATCOVER_PROJ_DATA'):
    pyproj.datadir.set_data_dir(os.environ['PEATCOVER_PROJ_DATA'])

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('PEATCOVER_ROOT', 'path/to/PEATLAND'))

# ---- inputs (read only)
LABELS = ROOT / 'Training_database/training_labels'
PRED_DB = {'Europe': ROOT / 'Prediction_Database/Europe',          # 17-variable stacks, one .mat per 5 deg tile
           'NA': ROOT / 'Prediction_Database/NA',
           'Dateline': ROOT / 'Prediction_Database/Dateline'}
FEATURES17 = ['dist0005', 'dist0100', 'dist1000', 'hand0005', 'hand0100', 'hand1000', 'merit_slope', 'elevation',
              'wtd', 'ls_b1', 'ls_b2', 'ls_b3', 'ls_b4', 'ls_b5', 'ls_b6', 'ls_b7', 'ndvi']
OPERA_NAMES = ['opera_iv', 'opera_ow']           # inundated vegetation / open water occurrence (%), 2025

# input sets (columns of the 19-column label / point matrices; column 8 = WTD, 17 = OPERA inundated vegetation,
# 18 = OPERA open water). A model's name starts with its input set: f17_..., f17iv_...
INPUT_SETS = {'17': list(range(17)),                           # 17 inputs
              '17iv': list(range(17)) + [17]}                  # + OPERA inundated vegetation (18 inputs)


def input_cols(name):
    """columns used by the model 'name' (its input set is the end of the first part of the name: f17_..., f17iv_...)"""
    import re
    return INPUT_SETS[re.search(r'(\d\d(?:iv|ow)?)$', name.split('_')[0]).group(1)]

# ---- OPERA DSWx-S1 occurrence 2025 (Isik et al., 2026, 30 m)
OPERA_SRC = {'opera_iv': ROOT / 'Peat_Wetland_Dataset/2_Classification/OPERA_DSWx-S1_occurrence_Isik2026/'
                             'inundated.vegetation_dswx.s1_30m_p_20250101_20251231_go_epsg.4326_v20260425.tif',
             'opera_ow': ROOT / 'Peat_Wetland_Dataset/2_Classification/OPERA_DSWx-S1_occurrence_Isik2026/'
                             'open.water_dswx.s1_30m_p_20250101_20251231_go_epsg.4326_v20260425.tif'}

# ---- outputs
WORK = Path(os.environ.get('PEATCOVER_WORK', str(ROOT / 'Processed/PeatCover_work')))                         # large intermediate files
OPERA90 = ROOT / 'Processed/OPERA_DSWx_S1_90m_tilegrid'               # 90 m OPERA on the tile grid
TAB = HERE.parent / 'results' / 'tables'
FIG = HERE.parent / 'results' / 'figures'
LOG = HERE.parent / 'results' / 'logs'
for d in (WORK, OPERA90, TAB, FIG, LOG):
    d.mkdir(parents=True, exist_ok=True)

# ---- tile grid: 5 x 5 deg tiles + 20 px buffer, 6040 x 6040 px of 1/1200 deg (3 arc-sec),
#      pixel centres on multiples of 1/1200 deg; stacks are stored column-major (MATLAB): index = row + col * 6040
N = 6040
RES = 1 / 1200
BUF = 20


def tile_origin(tile):
    """lower-left corner (lat, lon) in degrees of a tile name such as N60E025 / N60W175 / S15E045."""
    lat = int(tile[1:3]) * (1 if tile[0] == 'N' else -1)
    lon = int(tile[4:7]) * (1 if tile[3] == 'E' else -1)
    return lat, lon


def tile_centres(tile):
    """pixel-centre latitude (rows, north -> south) and longitude (columns, west -> east) of a tile stack."""
    lat0, lon0 = tile_origin(tile)
    i = np.arange(N)
    lat = lat0 + 5 + (BUF - 1 - i) * RES        # row 0 = lat0 + 5 + 19/1200
    lon = lon0 + (i - (BUF - 1)) * RES          # col 0 = lon0 - 19/1200
    return lat, lon


def tile_transform(tile):
    """affine transform (pixel edges) of the tile grid, for writing GeoTIFFs."""
    from rasterio.transform import from_origin
    lat, lon = tile_centres(tile)
    return from_origin(lon[0] - RES / 2, lat[0] + RES / 2, RES, RES)


def all_tiles():
    """{tile: region} for every prediction stack."""
    out = {}
    for reg, d in PRED_DB.items():
        for f in sorted(d.glob('*_predictionData.mat')):
            out.setdefault(f.name.split('_')[0], reg)
    return out


def stack_path(tile, region=None):
    region = region or all_tiles()[tile]
    return PRED_DB[region] / f'{tile}_predictionData.mat'


def core_tile_of(lat, lon):
    """as tile_of, but always the tile whose 6000 x 6000 CORE contains the point's pixel, and the (row, col) in the full tile
    (core = BUF .. BUF + 5999). tile_of puts points within half a pixel of a tile's west or north edge (e.g. every point at an
    exact multiple of 5 deg longitude) in the buffer (col / row 19) of the next tile: fine for the input stacks (the buffer is
    the same place), wrong for maps that store only the core (index -1 wraps to the other side of the tile).
    Pixel centres are at k/1200 deg; the core of tile (lat0, lon0) holds columns k = lon0*1200 + 1 .. lon0*1200 + 6000 and
    rows k = lat0*1200 .. lat0*1200 + 5999."""
    lat = np.asarray(lat, dtype='float64')
    lon = np.asarray(lon, dtype='float64')
    kx = np.rint(lon * 1200).astype(np.int64)
    ky = np.rint(lat * 1200).astype(np.int64)
    kx = np.where(kx <= -175 * 1200, kx + 360 * 1200, kx)    # -180..-175 deg = the E180 tiles of the 180 deg strip
    lon0 = (np.floor((kx - 1) / 6000) * 5).astype(int)
    lat0 = (np.floor(ky / 6000) * 5).astype(int)
    col = (kx - lon0 * 1200 - 1 + BUF).astype(int)
    row = ((lat0 + 5) * 1200 - 1 - ky + BUF).astype(int)
    name = np.array([f'{"N" if a >= 0 else "S"}{abs(int(a)):02d}{"E" if b >= 0 else "W"}{abs(int(b)):03d}'
                     for a, b in zip(lat0, lon0)])
    return name, row, col


# ---- spatial blocks of 2 x 4 deg (fixed random table)
NFOLD = 5


def block_fold(lat, lon, dlat=2.0, dlon=4.0):
    """fold id (0-4) of the ~200 km block (2 deg lat x 4 deg lon) that contains each point. The fold of a block comes
    from a fixed random table, so every data set (labels, field points) gets the same fold for the same block."""
    i = np.floor((np.asarray(lat) + 90) / dlat).astype(int)
    j = np.floor((np.asarray(lon) + 180) / dlon).astype(int)
    table = np.random.default_rng(42).integers(0, NFOLD, (int(180 / dlat) + 1, int(360 / dlon) + 1))
    return table[i, j]


def tile_of(lat, lon):
    """name of the 5 deg tile whose core contains each point, and the (row, col) of the point in that tile stack."""
    lat = np.asarray(lat, dtype='float64')
    lon = np.asarray(lon, dtype='float64')
    lon = np.where(lon < -175, lon + 360, lon)          # -180..-175 deg = the E180 tiles of the 180 deg strip (stacks on 180..185)
    lat0 = np.floor(lat / 5) * 5
    lon0 = np.floor(lon / 5) * 5
    lon0 = np.where(lon == 180, 175, lon0)              # the 180 deg meridian itself: last core column of the E175 tiles
    row = np.rint((lat0 + 5) * 1200 + (BUF - 1) - lat * 1200).astype(int)
    col = np.rint(lon * 1200 - lon0 * 1200 + (BUF - 1)).astype(int)
    name = np.array([f'{"N" if a >= 0 else "S"}{abs(int(a)):02d}{"E" if b >= 0 else "W"}{abs(int(b)):03d}'
                     for a, b in zip(lat0, lon0)])
    return name, row, col


def opera_at(lat, lon):
    """OPERA inundated-vegetation and open-water occurrence (%) of the 90 m cell containing each point (NaN where
    the tile is missing or outside the OPERA coverage)."""
    import rasterio
    name, row, col = tile_of(lat, lon)
    out = np.full((len(name), 2), np.nan, dtype='float32')
    for t in np.unique(name):
        f = OPERA90 / f'{t}_opera.tif'
        if not f.exists():
            continue
        k = np.where(name == t)[0]
        with rasterio.open(f) as r:
            a = r.read()
        v = a[:, row[k], col[k]].T.astype('float32')
        v[v == 255] = np.nan
        out[k] = v
    return out

# ---- data of the repository
REPO = HERE.parent
CANADA_SPLIT = REPO / 'data/networks/canada_tile_split.txt'      # training / test tiles of the Canadian regression network

