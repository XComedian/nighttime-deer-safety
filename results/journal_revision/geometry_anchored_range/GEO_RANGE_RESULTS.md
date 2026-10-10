# Geometry-anchored range (G0/G1) + ZOD species tagging (2026-10-03)

## (b) Species tags — `output/zod/species_tags.csv`
178 candidates (large rural/highway animals + any animal >= 1.2 m), tagged visually from full-resolution crops:
78 live hoofed (26 sheep, 19 horse, 19 reindeer, 8 cattle, 6 deer-or-reindeer), 22 statues/sculptures
(ZOD "Animal" includes moose/horse/lion statues — excluded), 11 dogs, 67 unidentifiable.
Hoofed: 18 night/twilight, 30 at 30–60 m.

## (c) Method
Z = f * H / h_box. f = focal (px); h_box = box height (px); H = animal box height (m):
- G0 constant H = 1.41 m (median rendered-deer H* = Z * h_box / f);
- G1 HeightNet: 4-block CNN on the box crop resized to 64x64 + log aspect ratio -> log H; scale- and
  camera-blind input, so all camera dependence is in the equation.
Rendered focal lengths were never logged; recovered per frame from the paved-surface mask width x depth
(`recover_synthetic_focal.py`): all 6,000 frames, flat histogram over the 24–50 mm prior, p99/p1 = 2.05 vs 2.08,
road width 7.10 m. Scripts: `geo_range.py` (LORO training), `geo_eval.py` (same detections),
`geo_real_eval.py` (ZOD + dashcam).

## Rendered roads, leave-one-road-out, identical detections (mean of 10 roads)
| Band | Joint: range head | G0 | G1 | Roads >= 0.60 (head -> G1) | MAE head -> G1 |
|---|---|---|---|---|---|
| 30–40 m | 0.761 | 0.817 | 0.849 | 7 -> 9 | 3.15 -> 2.27 m |
| 40–50 m | 0.679 | 0.637 | 0.734 | 7 -> 7 | 3.58 -> 3.01 m |
| 50–60 m | 0.594 | 0.496 | 0.594 | 5 -> 5 | 3.85 -> 4.49 m |
- Unmarked Rural Farmland: head 0.40/0.43/0.43 (MAE 7.4 m) -> G1 0.88/0.84/0.71 (MAE 2.4–3.5 m): the
  lane-marking dependence disappears.
- Weak spot: Rural Crest 50–60 m (G1 MAE 10.9 m) — the crest hides legs, boxes too short -> overestimate.

## ZOD real (LiDAR ground truth; G1 trained on renders only)
| Group (54 deg view / 70 deg view) | Head rel. err / elasticity | G1 rel. err / elasticity | Size oracle rel. err |
|---|---|---|---|
| Hoofed, n=52/78 | 0.42 / −0.01 ; 0.57 / −0.14 | 0.19 / 0.89 ; 0.23 / 0.95 | 0.12 ; 0.10 |
| Deer-like (reindeer/deer), n=24/25 | 0.33 / −0.18 ; 0.27 / 0.06 | 0.23 / 0.86 ; 0.23 / 0.82 | 0.16 ; 0.15 |
| Horse/cattle, n=16/27 | 0.45 ; 0.72 | 0.23 ; 0.23 (G0 better: 0.16–0.17) | 0.12 |
| Sheep, n=12/26 | 0.57 ; 0.65 | 0.14 ; 0.23 | 0.10 ; 0.09 |
| Dogs, n=8/11 | 0.48 | 0.55–0.65 (height overestimated) | 0.06 |
(rel. err = median |error| / true range)
- G1 estimates real hoofed-animal height well from renders alone: median 1.10 m vs true 1.04 m; it
  underestimates horses/cattle (1.15 vs 1.5–1.6 m) and overestimates dogs (1.16 vs 0.75 m).
- Dashcam clips (unknown focal): elasticity 0.85 (head 0.19); estimated height varies 6% (CV) along a track.

## Findings
1. Anchoring range in camera geometry fixes the real-world failure: range now scales correctly with true
   distance on real images (elasticity 0.82–0.95 vs ~0) and is camera-aware by construction.
2. On renders it matches or beats the learned head and removes the lane-marking dependence.
3. Residual real error is dominated by animal-size variation (~20–23% for hoofed animals vs ~10–15% with true
   height): at 50 m that is about ±10 m, still above the 5 m project tolerance. Range accuracy is now bounded by
   how well the animal's size is known, not by the camera.
4. Class matters: a deer-size prior is wrong for dogs and for large livestock; a deployed system needs the
   species (or at least size class) from the detector.

## Caveats
- ZOD subsets are small (24–78 animals), mostly daytime, European livestock/reindeer rather than white-tailed deer.
- Evaluated at ground-truth boxes on ZOD (the synthetic-trained detector finds ~1% of real animals); a real
  pipeline would pair this range with a real-trained detector.
- Detected-box height errors grow at range (15 px boxes at 50–60 m) and with occlusion (crest).
