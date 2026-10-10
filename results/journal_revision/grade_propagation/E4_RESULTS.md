# E4 — Predicted grade in the stopping-distance screen (2026-10-02)

Script: `scripts/e4_grade_ssd.py` → `frame_grades.csv`, `summary.json`.
Data: nuScenes night, held-out test log (98 usable frames, 3 scenes) and validation log (284 frames, 13 scenes);
frames need >= 50 m of future trajectory. Models: synthetic-only geometry checkpoint and real-night adapted heads.

## Design
- Reference grade = distance-weighted mean vehicle pitch (localization attitude) over the next 0–60 m actually driven.
- `imu_only` = current vehicle pitch (gravity-referenced attitude, assumes constant grade ahead).
- `imu_X` = current pitch + relative road tilt ahead from X: LiDAR plane (sensor reference) or the geometry model's
  predicted normals / predicted depth in a fixed lower-image ROI (camera-only; no LiDAR used for pixel selection).
- Propagation: AASHTO SSD with t = 2.5 + 0.30 s, f = 0.35; boundary = max speed with SSD <= 60 m.
- Why a gravity reference: a camera only sees the road relative to itself; on a constant grade the vehicle is tilted
  with the road, so absolute grade needs vehicle pitch (IMU) or a map. The camera can only add grade *change* ahead.

## Results (test | val)

| Estimator | Grade MAE (%) | SSD MAE @30 mph (m) | SSD p95 @30 mph (m) | 60 m boundary MAE (mph) |
|---|---|---|---|---|
| Pitch only | 1.19 \| 1.48 | 1.05 \| 1.37 | 2.82 \| 4.13 | 0.30 \| 0.38 |
| Pitch + LiDAR plane | 1.02 \| 1.08 | 0.85 \| 1.00 | 2.76 \| 3.42 | 0.25 \| 0.28 |
| Pitch + adapted normals | 1.37 \| 1.81 | 1.16 \| 1.59 | 3.34 \| 4.60 | 0.34 \| 0.46 |
| Pitch + adapted depth | 2.45 \| 2.43 | 1.89 \| 1.97 | 4.42 \| 4.89 | 0.58 \| 0.59 |
| Pitch + synthetic-only normals | 30.1 \| 30.3 | 12.7 \| 13.2 | 16.7 \| 19.8 | 5.08 \| 5.17 |
| Pitch + synthetic-only depth | 14.0 \| 14.9 | 7.8 \| 8.7 | 11.8 \| 14.2 | 2.79 \| 3.01 |

Relative tilt vs LiDAR plane: adapted normals 0.92 | 1.47 % MAE; adapted depth 2.01 | 2.14 %; synthetic-only 13–30 % (biased).
Reference path grade: test mean −1.2 % (p5–p95 −4.0 to +2.0); val mean −1.5 % (−9.2 to +2.4).

## Findings
1. **Grade error barely moves the screen at these speeds.** Even the camera-depth route changes SSD by < 2 m (MAE) at
   30 mph and the 60 m speed boundary by < 0.6 mph. Analytically dSSD/dG ≈ 0.75 m per 1 % grade at 30 mph, because
   the braking term (the only grade term) is the minor part of SSD below ~30 mph. Range, not grade, sets the boundary.
2. **The camera adds nothing over vehicle pitch on these roads.** Adapted normals match the LiDAR plane within ~1 %
   but do not predict the 0–60 m path grade better than current pitch alone; only the LiDAR plane improves slightly.
3. **Synthetic-only geometry is unusable for real grade** (13–30 % bias) — real-night adaptation is required.
   The synthetic normal bias is consistent with the normals-frame issue below.

## Caveats
- Reference = localization attitude, includes suspension/braking pitch; not surveyed grade. Singapore night roads are
  mostly gentle (|G| < 4 % for 90 % of test frames), so steep-grade behavior is untested.
- One test log; val log was used for head checkpoint selection (on AbsRel + normal error, not grade).

## Side finding — synthetic normals are not camera-frame
`generate_deer_crossing_nightroad.py` calls `bproc.renderer.enable_normals_output` (line 553) before the frame range
(564) and camera poses (710) are set. BlenderProc bakes the per-frame camera rotation into the normals at that call,
so every frame uses one fixed rotation: the stored normals are world-frame normals under a constant rotation, not
camera-frame normals. Evidence: the road-normal z-component is nearly constant across frames despite randomized pitch,
and differs between scenes in proportion to authored grade. Paper text calling them "camera-frame normals" must change.
