# Nighttime Deer Safety Pipeline

Research code and supporting artifacts for a monocular nighttime deer-safety
study by Ningning Xu, Yunxiang Yang, and Jidong Yang at the University of
Georgia.

Repository: <https://github.com/XComedian/nighttime-deer-safety>

The project studies a monocular nighttime pipeline that combines deer detection,
metric range estimation, road-geometry estimation, and a warning-only
stopping-distance advisory. The primary evaluation holds out complete rendered
road scenes rather than randomly splitting neighboring frames.

## Repository scope

This repository contains:

- the model architecture and experiment training/evaluation scripts;
- scene metadata and the exact experiment split definitions;
- compact machine-readable experiment outputs; and
- the advisory-controller replay input and unit tests.

It intentionally does **not** contain the 66 GB synthetic dataset, trained
checkpoints, nuScenes files, restricted real images, or unpublished manuscript
files.

## Data and checkpoints

The synthetic dataset and trained checkpoints will be hosted separately:

**Archive URL:** `DATA_ARCHIVE_URL_TO_ADD`

After downloading, arrange the files as:

```text
data/
├── synthetic_deer_dataset/
│   ├── 01_near_range/
│   ├── 02_range_30-60m/
│   ├── 03_furniture_ablation/
│   ├── SCENES.csv
│   └── splits/
└── checkpoints/
```

nuScenes data are not redistributed. Obtain the official trainval release from
the nuScenes provider and place or link it at `data/nuscenes/`.

## Environment

The experiments were run with Python 3.11 and the versions recorded in
`requirements.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA is required for practical training time. CPU execution is sufficient for
the advisory unit tests and compact analyses.

## Primary range experiment

Convert each archived scene into the scene-disjoint YOLO layout. For example:

```bash
python scripts/loso_convert.py \
  data/synthetic_deer_dataset/02_range_30-60m/rural_baseline c8
```

Repeat using the `original_id` to short-tag mapping in `metadata/SCENES.csv`,
then build the eleven outer folds:

```bash
python scripts/loso_build_folds.py \
  c7 c8 c9a c9b c9c c10 c11 c12 c13 c14 c15
```

Train one model per held-out scene. The example below holds out `c8`; its
training YAML contains only the other scenes and an inner validation split.

```bash
python scripts/train_distance_multitask.py \
  --data data/loso/folds/c8/data.yaml \
  --distances-json data/loso/folds/c8/train_distances.json \
  --project output/loso \
  --name c8 \
  --weights data/checkpoints/initialization.pt \
  --gpu 0
```

After all fold checkpoints are present:

```bash
LOSO_OUT=output/loso \
python scripts/loso_eval.py c7 c8 c9a c9b c9c c10 c11 c12 c13 c14 c15
```

The retained study values are preserved in
`results/loso/loso_results.json`. Rural aggregates use `c8`--`c15`; `c7` is
reported separately as the forest-domain probe.

## Real-night geometry experiment

The geometry protocol uses the official nuScenes trainval release, divides
night scenes by acquisition log, projects LiDAR into the front camera, and
adapts only the geometry heads:

```bash
python scripts/nuscenes_geometry_experiment.py \
  --root data/nuscenes \
  --protocol full-night \
  --mode all \
  --device cuda:0 \
  --out output/nuscenes_geometry_full
```

The synthetic geometry checkpoint expected by this script can be supplied at
`output/synthetic/searched_geo/weights/last.pt`. See `--help` for other
arguments and `results/geometry/` for the retained summaries.

## Advisory replay

The compact input needed to reproduce the historical controller stress test is
included, so this stage does not require images or checkpoints:

```bash
python scripts/replay_dynamic_ssd.py \
  --input results/advisory/replay_input.csv \
  --out output/dynamic_ssd
```

Run the controller tests with:

```bash
pytest -q scripts/test_dynamic_ssd_advisory.py
```

The replay validates controller logic under declared assumptions; it is not an
on-road deployment or automatic-braking validation.

## Reproducibility notes

`REPRODUCIBILITY.md` maps the principal study claims to retained artifacts.
The released rendered frames are fixed experimental objects. The original
render scripts did not record every random seed, so bitwise regeneration of the
synthetic dataset is not claimed.

## Licenses

Code is released under the MIT License. Metadata and result tables are released
under CC BY 4.0; see `DATA_ACCESS.md`. The separately hosted dataset will carry
its own access terms after third-party asset-license review. nuScenes and other
third-party content remain governed by their original licenses.

## Safety notice

This is research software. It is not certified for vehicle control, collision
avoidance, or public-road deployment.
