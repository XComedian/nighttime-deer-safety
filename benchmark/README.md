# Animal Range Benchmark (nighttime deer warning)

Benchmark accompanying *Reliability-Based Screening of Camera-Only Nighttime Deer Warning with Geometry-Anchored Range*.
It tests whether a monocular per-animal range estimator **scales correctly with true distance on real cameras**, not
only whether it is accurate on its training distribution.

## Contents

```
synthetic/
  focal_px.json               per-frame focal length (px) recovered for the 6,000 rural range frames
  focal_recovery_report.json  recovery check (uniform over the 24-50 mm lens prior; road width 7.10 m)
  loro_splits.json            leave-one-road-out splits over the ten rural range roads
zod_animal_range/
  annotations.json            2,191 ZOD Frames animals with LiDAR 3D boxes (derived annotations, no images)
  annotations_summary.csv     one row per animal
tools/
  make_views.py               regenerate the 960x540 undistorted pinhole views (54.4 and 70 deg) from a ZOD download
  evaluate.py                 score per-animal range predictions
```

The rendered images, depth, normals, and instance masks are the 23-scene synthetic nighttime deer corpus
(`synthetic_deer_dataset/`, see its README); the files above add the camera intrinsics and splits needed for
range evaluation.

## ZOD annotations

Each record: `frame_id`, `annotation_uuid`, `depth_cam_m` (camera-frame forward depth of the LiDAR box center),
`center_cam_m`, `size_lwh_m`, `box_fisheye_xyxy` (original 3848x2168 image), and for each virtual view
`view54.4` / `view70`: `in_view`, `box_xyxy`, `fx`, `cx`, `cy` (960x540 pinhole on the same optical axis).
`review_tag` is the visual species review of 178 large or tall animals: `ungulate` (78: 26 sheep, 19 horses,
19 reindeer, 8 cattle, 6 deer-or-reindeer, see `species`), `dog` (11), `statue` (22; ZOD labels some statues as
Animal, exclude them), `unclear` (67); all other animals are `not_reviewed`.

## Metrics (`tools/evaluate.py`)

Predictions CSV: `frame_id, annotation_uuid, view, pred_m`.
- median absolute relative error with a frame-clustered bootstrap 95% interval (animals in a herd share a frame);
- share within 10% of LiDAR;
- scaling slope: regression of log predicted on log true range (1 = scales correctly; 0 = ignores distance);
- zoom test: median ratio of predictions 70/54.4 deg for the same animals (camera-aware estimate ~1).

Reference results (hoofed, 70-degree view): learned range head 57% / slope -0.14; geometry-anchored G1 23% / 0.95;
Metric3D v2 ViT-L 20% / 0.75; size oracle 10% / 0.98.

## License

ZOD (Zenseact Open Dataset) is released under CC BY-SA 4.0; the derived annotations and regenerated views keep that
license and must credit Zenseact (Alibeigi et al., ICCV 2023). ZOD images are not redistributed here; obtain ZOD
Frames from https://zod.zenseact.com.
