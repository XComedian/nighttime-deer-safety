# E5 — Real-world range evidence without new collection (2026-10-02)

## E5a Scale-free consistency on the four real dusk clips  (DONE)
Scripts: `scripts/e5_real_consistency.py` (real tracks), `scripts/e5_synthetic_control.py` (synthetic control).
Outputs: `summary.json`, `tracks_primary.csv`, `synthetic_control.json`.

Why scale-free: clips are 724x470 re-encoded video with unknown focal length/distortion, curved rolling roads, and
double-solid center lines (no MUTCD dash ruler), so no metric distance reference can be built.
Test: for one deer at fixed focal length, Z * h_box = f * H is constant, so within a track
d log Z / d log h = -1. The negative within-track slope therefore estimates the range head's *elasticity*
(d log predicted / d log true): 1 = correct relative ranging, 0 = output ignores distance.
Range sampled at user-labelled box centers (isolates range from detection). 10 tracks, 180 boxes; 8 tracks with
box-height ratio >= 1.5 are informative.

| Setting | Elasticity |
|---|---|
| Held-out synthetic roads (pred vs GT): Rural Straight / Autumn / Conifer | 0.94 / 0.90 / 0.90 |
| Held-out synthetic: Rural Baseline | 0.51 |
| Real clips, pooled within-track, Rural Baseline checkpoint | 0.21 |
| Real clips, pooled within-track, range over 10 rural checkpoints | 0.13–0.22 (mean 0.19) |

- Longest tracks: box height grows 4.7x and 6.7x (true range shrinks by the same factor); predicted range changes
  1.8x and 2.0x (e.g. 58 -> 33 m, 60 -> 29 m). Label noise (~1-2 px) cannot explain this at a 6.7x size change.
- Real predictions stay inside 29.5–51.6 m (p5–p95) — the 30–60 m training support. Size changes imply some deer
  came well inside 30 m, where the head (trained on 30–60 m roads) cannot output; tracks whose size change is
  compatible with staying in 30–60 m (ratio ~2) give slopes −0.38 to +0.26, also far from −1.
- Absolute level: median Z*h implies a 1.35 m box height at a 70-degree HFOV (0.90 m at 50, 1.92 m at 90) —
  plausible, but this only shows the output sits at a plausible mean, not that it tracks distance.

**Finding:** on real dusk footage the synthetic-trained range head barely responds to distance (elasticity ~0.2 vs
~0.9 on unseen synthetic roads); it reverts toward its training-range prior. No real-world range claim is supportable;
real-road evidence is detection-only. Causes cannot be separated here: synthetic-to-real appearance gap, real deer
outside the 30–60 m support, unknown camera intrinsics.

## E5b Real metric range on deer-scale animals  (BLOCKED on data access)
- nuScenes (local): 52 animal instances, median height 0.6 m (dogs/cats), 59 night annotations -> not usable.
- **Zenseact Open Dataset (ZOD)** is the only candidate found: Animal class with LiDAR 3D boxes to 245 m,
  ~19% night (~20k frames), ~15% rural roads, 14 European countries incl. northern Sweden (moose/deer/reindeer
  plausible), CC BY-SA 4.0, front camera 3848x2168, 120 deg fisheye (needs undistortion to pinhole crops).
  Animal count is not stated in the paper text — must be checked from annotations after access.
  Access: email opendataset@zenseact.com (name, affiliation, Dropbox-linked email, intended use); then
  `pip install "zod[cli]"` and `zod download` (annotations-only streams possible via --no-* flags).
