# nuScenes Night Geometry Experiment

Acquisition-log-disjoint nighttime protocol; the final test log contains five scenes.

| Metric | Synthetic only | Adapted | Relative change |
|---|---:|---:|---:|
| Depth AbsRel | 0.462 | 0.297 | +35.7% |
| Depth RMSE (m) | 16.726 | 9.134 | +45.4% |
| Depth delta < 1.25 | 0.128 | 0.586 | +358.3% |
| Normal error (degrees) | 12.665 | 1.495 | +88.2% |

Intervals use a paired scene-cluster bootstrap over the five test scenes; they remain conditional on one acquisition log.

Depth AbsRel improved in 4/5 test scenes; scene-level heterogeneity is retained in `test_scene_metrics.csv`.

Normals are LiDAR-derived RANSAC road-plane pseudo-references. Grade is a local road-to-ego tilt proxy; no absolute surveyed road grade is evaluated.

Figure: `output/nuscenes_geometry_full/nuscenes_real_geometry_comparison.png`
