# Reproducibility Map

This file maps the study's principal quantitative findings to the retained
machine-readable artifacts.

| Study evidence | Retained artifact |
|---|---|
| Eleven-scene leave-one-scene-out detection and range results | `results/loso/loso_results.json` |
| Rural per-band means and variation | `results/loso/loso_summary.md` |
| Road-unit bootstrap intervals and pass rates | `results/loso_road_bootstrap/road_bootstrap.json` |
| Forest environment-recovery experiment | `results/forest/loso_results.json` |
| Lane-marking and delineator-post ablation | `results/ablation/cue_results.json` |
| nuScenes baseline and adapted test metrics | `results/geometry/*_test_summary.json` |
| Per-scene geometry metrics | `results/geometry/test_scene_metrics.csv` |
| Scene-cluster bootstrap intervals | `results/geometry/paired_frame_intervals.json` |
| Advisory state distributions and transition audit | `results/advisory/metrics.json` |
| Advisory sensitivity analysis | `results/advisory/sensitivity.csv` |
| Historical advisory replay input | `results/advisory/replay_input.csv` |
| Final-checkpoint 60 m replay input, metrics, and summary | `results/advisory_loso60/` |
| Single-scene affine recalibration probe | `results/recalibration/recalibration_table.csv` |
| Dusk negative-frame false-positive audit | `results/real_detection/summary.json` |
| Stopping-distance and reaction-time scenarios | `results/ssd/*.csv` |

## Validation conventions

- Outer evaluation units are complete scenes, not randomly selected frames.
- Detector matches use IoU at least 0.50 and confidence at least 0.25.
- Distance bands are 30--40, 40--50, and 50--60 m.
- Joint success requires a matched detection with absolute range error no more
  than 5 m.
- The rural aggregate contains the ten open-rural folds (`c8`--`c15`).
- Road-resampling intervals use the held-out rural road as the sampling unit
  and do not represent model-training seed variation.
- The held-out forest scene (`c7`) is reported separately because its environment
  is absent from the corresponding training folds.
- The nuScenes test scenes belong to one held-out acquisition log. Reported
  resampling intervals describe within-log variation, not across-route
  generalization.

## Known reproduction boundary

The scripts and fixed artifacts reproduce the evaluation and analysis path. The
original Blender render scripts did not record all pseudorandom seeds; therefore,
bitwise regeneration of the released synthetic frames is outside the
reproducibility claim.

The rendered range corpus ends at 60 m. Accordingly, 60 m is an evaluation
ceiling rather than an estimated maximum detection distance; performance beyond
the corpus ceiling is untested.
