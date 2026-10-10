# Nighttime Deer Safety Pipeline

Research code and supporting artifacts for a monocular nighttime deer-safety
study by Ningning Xu, Yunxiang Yang, and Jidong Yang at the University of
Georgia.

Repository: <https://github.com/XComedian/nighttime-deer-safety>

The project studies monocular nighttime components for deer detection, metric
range estimation, road-geometry estimation, and stopping-distance safety
screening. The components are evaluated separately; this repository does not
claim an integrated warning system. The primary evaluation holds out complete
rendered road scenes rather than randomly splitting neighboring frames.

## Repository scope

This repository contains:

- the model architecture and experiment training/evaluation scripts;
- scene metadata and the exact experiment split definitions;
- compact machine-readable experiment outputs; and
- the advisory-controller replay input and unit tests.

The selected network, architecture-search protocol, and task-head definitions
are documented in [`docs/MODEL_ARCHITECTURE.md`](docs/MODEL_ARCHITECTURE.md);
the exact detector YAML is in
[`configs/detection_range_model.yaml`](configs/detection_range_model.yaml).

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

Reproduce the road-unit uncertainty intervals with:

```bash
python scripts/bootstrap_loso_roads.py \
  --input results/loso/loso_results.json \
  --output-dir output/loso_road_bootstrap \
  --replicates 100000 \
  --seed 20260728
```

The retained road-bootstrap table and summary are in
`results/loso_road_bootstrap/`. These intervals describe heterogeneity across
the ten represented rural roads; they do not include model-training seed
variation.

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

## Corrected 60 m advisory replay

The compact prediction input from the final crossing8 leave-one-scene-out
checkpoint is included, so the retained 60 m controller replay does not require
images or a checkpoint:

```bash
python scripts/replay_dynamic_ssd_loso60.py \
  --input results/advisory_loso60/c8_loso_predictions.csv \
  --output-dir output/dynamic_ssd_loso60
```

Its retained metrics and summary are in `results/advisory_loso60/`. Predicted
mode uses point estimates with zero uncertainty because no independently
calibrated uncertainty model exists. The input includes matched detections only,
so nuisance alarms from unmatched false positives are not measured.

With the archived checkpoint and converted crossing8 scene, regenerate the
prediction input using:

```bash
python scripts/export_loso_predictions.py \
  --scene data/loso/scenes/c8 \
  --weights data/checkpoints/c8_loso_best.pt \
  --output output/dynamic_ssd_loso60/c8_loso_predictions.csv \
  --device cuda:0
```

The earlier 40 m diagnostic artifacts remain in `results/advisory/` for
provenance but are not the controller evidence used by the submitted paper.

Run the controller tests with:

```bash
pytest -q scripts/test_dynamic_ssd_advisory.py
```

The replay validates controller logic under declared assumptions; it is not an
on-road deployment or automatic-braking validation.

## Journal revision (October 2026): geometry-anchored range, reliability screening, benchmark

The revised study (*Reliability-Based Screening of Camera-Only Nighttime Deer Warning with Geometry-Anchored
Range: Synthetic Training and Real-World Evaluation*) adds the following. Scripts are in `scripts/`; compact
outputs are in `results/journal_revision/`.

- **Recovered camera intrinsics** for the rendered rural range roads (the lens was randomized but not logged):
  `scripts/recover_synthetic_focal.py`, results in `benchmark/synthetic/`.
- **Geometry-anchored range** `Z = f * H / h`, the distance decomposition of MonoRCNN/GUPNet applied to animals,
  with a crop-based height network (HeightNet) trained on rendered deer only: `geo_range.py` (leave-one-road-out
  training), `geo_eval.py` (identical detections vs. the learned range head), `geo_paper_stats.py`,
  `geo_cue_eval.py` (road-furniture scenes), `geo_real_eval.py` (ZOD and dusk clips).
- **Real-world range on ZOD** (LiDAR ground truth, Zenseact Open Dataset): `zod_fetch_images.py` (filtered download;
  needs `ZOD_URL` from Zenseact), `zod_range_eval.py` (virtual pinhole views), `zod_bootstrap_ci.py`,
  `species_prior_eval.py`, `zod_finetune.py`, `species_data.py` / `species_classifier.py`.
- **Depth foundation-model baselines** (Metric3D v2, UniDepthV2, Depth Anything V2): `depth_baselines.py`,
  `summarize_depth_baselines.py`, `head_gt_boxes.py`.
- **Grade propagation, real-footage consistency, warning decisions**: `e4_grade_ssd.py`, `e5_real_consistency.py`,
  `e5_synthetic_control.py`, `e6_decision_eval.py`, `e6_geo.py`.
- **Reliability-based stopping-distance screening** (Monte Carlo probability of late warning with measured driver
  response): `reliability_ssd.py`, `reliability_crossfit.py`.
- Figures and tables: `make_paper_tables.py`, `make_geo_figures.py`, `make_reliability_figure.py`, `make_zod_vis.py`.

Data locations are read from environment variables with repository-relative defaults: `ZOD_ROOT` (default
`data/zod`), `SYNTHETIC_ROOT`, `COCO_ROOT`, `REAL_DEER_ROOT`.

### Animal range benchmark

`benchmark/` contains derived ZOD animal-range annotations (2,191 animals with LiDAR boxes, a species review of
178, undistorted boxes and intrinsics for two virtual views), the rendered-road focal lengths and splits, and
`tools/evaluate.py` (median relative error with frame-clustered intervals, scaling slope, zoom test). See
`benchmark/README.md`. The ZOD-derived files are CC BY-SA 4.0 (`benchmark/zod_animal_range/LICENSE.md`).

### Correction to the retained LOSO values

`results/loso/loso_results.json` holds the July 23, 2026 re-evaluation used in the TRB submission (50-60 m joint
mean 0.623). Re-evaluating the same frozen checkpoints with the current `loso_eval.py` does not reproduce it and
instead matches the original run in `results/journal_revision/loso_reproduced/` (50-60 m joint mean 0.594;
detection recall is identical in both). The journal revision uses the reproducible values; the earlier file is kept
for provenance.

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
