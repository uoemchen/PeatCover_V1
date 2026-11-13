# PeatCover

Code and data of PeatCover, a 90 m peatland map of the northern high latitudes:

> Chen, M., Ciais, P., Hugelius, G., and Aires, F.: PeatCover: Towards a 90 m peatland map of the northern high latitudes
> using satellite remote sensing and a priori inventories, Earth Syst. Sci. Data Discuss., https://doi.org/10.5194/essd-2026-254, 2026.

The maps (binary peatland map and Peatland Index, GeoTIFF) are on Zenodo: https://doi.org/10.5281/zenodo.21031869.

## Contents

| Folder | Content |
|---|---|
| `1_inputs/` | The 18 inputs on the 5 x 5 deg tile grid (Landsat 8, MERIT DEM / Hydro, water table depth, OPERA inundated vegetation) and the reference peat fraction of each tile |
| `2_labels/` | Training labels from the a priori inventories, training data of the Canadian regression network |
| `3_training/` | Training of the networks (MATLAB) |
| `4_mapping/` | Peatland Index, downscaling and binary maps of Eurasia and North America |
| `5_evaluation/` | Cross-validation and map accuracy |
| `data/` | Training labels, evaluation points, per-tile thresholds and trained networks |

The header of each script describes its inputs, outputs and options.

## Data

- `data/training_labels/`: location, label (1 peat, 0 non-peat) and split of the training pixels.
- `data/evaluation/`: field observations used for the map accuracy (Peat-DBase, BRO-BHR-P, PDD-Can-1), folds of the
  spatial cross-validation and the points of the PI distributions. The UK LandIS borings are not included (licence).
- `data/thresholds/`: reference peat fraction and PI threshold of each tile.
- `data/networks/`: trained networks (exported by `3_training/export_net.m`, applied by `4_mapping/matlab_net.py`).

The large inputs (per-tile input stacks, a priori inventories, satellite data) are not included; their sources are given
in the paper.

## Requirements

- Python: `conda env create -f environment.yml`
- MATLAB R2022b with the Deep Learning Toolbox and the Mapping Toolbox; `inpoly2` (https://github.com/dengwirda/inpoly)
- Environment variables: `PEATCOVER_ROOT` (data root), `PEATCOVER_WORK` (intermediate files), `MERIT_DIR`, `WTD_DIR`

## Example

Accuracy of the binary map from Zenodo with the field observations:

```bash
python 5_evaluation/05_map_accuracy.py --map PeatCover.tif
```

## Licence

Code: MIT (`LICENSE`). The data files keep the terms of their sources (PEATMAP, GPD, CORINE Land Cover, Peat-DBase,
BRO-BHR-P, PDD-Can-1); please cite them when you use the data. Citation: `CITATION.cff`.
