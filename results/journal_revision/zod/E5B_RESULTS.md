# E5b — Real metric range on ZOD animals (2026-10-03)

Scripts: `zod_fetch_images.py` (filtered download), `zod_range_eval.py` (virtual pinhole views + eval),
`zod_summarize.py`. Data kept: 3,092 blurred front images (2,492 animal frames + 600 animal-free
night/twilight rural/highway frames), 14 GB total incl. annotations. Archives deleted after filtering.

## Protocol
- ZOD front camera: 3848x2168 Kannala-Brandt fisheye, 120 deg HFOV. Each frame resampled to a virtual
  960x540 pinhole view on the same optical axis (54.4 deg = mid training focal; 70 deg also run).
- GT: LiDAR 3D box (Velodyne frame) -> camera frame via calibration; range = camera-frame forward depth Z
  (synthetic labels are z-depth). Check: 452/467 3D centers project inside their 2D boxes.
- 2D boxes undistorted into the virtual view. Model: Rural Baseline LOSO checkpoint (trained on 10 rural renders).
- Baselines: flat ground (camera height 1.53 m, estimated from 1,481 annotated object bottoms; horizon at
  principal row), size prior (H = 1.0 m), size oracle (true LiDAR height; upper bound for size-based ranging).
- Subset of interest: "rural large" = length >= 1.0 m or height >= 0.8 m on rural roads/highways
  (82 animals in the 54 deg view, 125 in the 70 deg view; 23–26 night/twilight).
  Visual check: mixed dogs, horses, reindeer, a few deer-like animals — NOT deer-only.

## Results — rural large animals (54 deg view | 70 deg view)

| Method | MAE overall (m) | Elasticity | MAE 30–60 m (m) |
|---|---|---|---|
| Range head (at GT box) | 50.3 \| 52.9 | −0.01 \| 0.05 | 12.0 \| 10.6 |
| Flat ground | 43.3 \| 55.9 | 1.10 \| 1.03 | 16.2 \| 33.1 |
| Size prior 1.0 m | 24.2 \| 27.5 | 0.77 \| 0.80 | 10.2 \| 13.8 |
| Size oracle (LiDAR height) | 8.6 \| 8.7 | 1.02 \| 1.00 | 5.4 \| 5.3 |

- Detection: recall 0.012 (54 deg) / 0.000 (70 deg) on rural large animals; joint success 0 in every band.
- False alarms: 54% / 52% of the 600 animal-free night/twilight rural frames contain >= 1 deer detection.
- **Zoom test:** for the same 1,165 animals, widening the view 54 -> 70 deg shrinks them 1.36x, but the head's
  median prediction is unchanged (38.6 m vs 38.6 m; ratio 1.004). On real images the head emits a near-constant
  range near its 30–60 m training prior, independent of apparent size.
- All animals (1,169 in view) and the day vs night/twilight splits show the same pattern.

## Findings
1. Confirms E5a with metric ground truth: the synthetic-trained range head does not transfer to real imagery
   (elasticity ~0; predictions ~constant ~39 m). Its 30–60 m MAE (~11 m) only reflects that its constant output
   happens to sit inside that band.
2. The synthetic-trained detector does not transfer to real animals (recall ~1%) and false-alarms on about half of
   real night frames.
3. Simple geometry beats the learned head on real data. A known animal height gives 5.3–5.4 m MAE at 30–60 m with
   correct scaling; even a crude 1.0 m prior scales correctly (0.8). Flat-ground ranging only works within ~30 m
   (sensitive to pitch and grade).
4. Implication for the paper/method: metric range should be geometry-anchored (detected box height + camera
   intrinsics + a class height prior or estimated animal height), with the learned component estimating height
   or correcting the geometry rather than regressing distance from appearance.

## Caveats
- Species mix (dogs, horses, reindeer, some deer); ~80% daytime; European roads; one checkpoint.
- Size-oracle uses true height from LiDAR — an upper bound, not a deployable method.
