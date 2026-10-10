#!/usr/bin/env python3
"""Assemble the release benchmark: ZOD animal-range annotations (derived, no images), synthetic split/focal files,
and the evaluation tool. Output: deer_dataset/animal_range_benchmark/."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from zod_range_eval import ZOD, lidar_to_cam, virtual_camera  # noqa: E402

import cv2  # noqa: E402

REL = ROOT / "benchmark"
VIEWS = (54.4, 70.0)


def zod_annotations() -> list[dict]:
    tags = {}
    for r in csv.DictReader((ROOT / "output/zod/species_tags.csv").open()):
        tags[(f"{int(r['frame']):06d}", round(float(r["gt_z"]), 2))] = (r["tag"], r["species"])
    frames = sorted({r["frame"] for r in csv.DictReader((ROOT / "output/zod/animal_annotations.csv").open())})
    out = []
    for fr in frames:
        fid = f"{int(fr):06d}"
        calib = json.loads((ZOD / fid / "calibration.json").read_text())
        meta = json.loads((ZOD / fid / "metadata.json").read_text())
        T = lidar_to_cam(calib)
        cams = {v: virtual_camera(calib, v)[:3] for v in VIEWS}
        for a in json.loads((ZOD / fid / "annotations/object_detection.json").read_text()):
            pr = a["properties"]
            if pr.get("class") != "Animal" or "location_3d" not in pr:
                continue
            c = T @ np.r_[pr["location_3d"]["coordinates"], 1.0]
            if c[2] < 3:
                continue
            pts = np.array(a["geometry"]["coordinates"], dtype=np.float64)
            rec = {"frame_id": fid, "annotation_uuid": pr["annotation_uuid"], "time_of_day": meta["time_of_day"],
                   "road_type": meta["road_type"], "country": meta["country_code"],
                   "depth_cam_m": round(float(c[2]), 3), "center_cam_m": [round(float(x), 3) for x in c[:3]],
                   "size_lwh_m": [pr["size_3d_length"], pr["size_3d_width"], pr["size_3d_height"]],
                   "occlusion": pr.get("occlusion_ratio"), "unclear": pr.get("unclear"),
                   "box_fisheye_xyxy": [round(float(v), 1) for v in (*pts.min(0), *pts.max(0))]}
            tag, species = tags.get((fid, round(float(c[2]), 2)), ("", ""))
            rec["review_tag"] = tag or "not_reviewed"         # ungulate / dog / statue / unclear (reviewed subset)
            rec["species"] = species
            for v in VIEWS:
                K, D, P = cams[v]
                und = cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2), K, D, R=np.eye(3), P=P).reshape(-1, 2)
                x0, y0 = und.min(0); x1, y1 = und.max(0)
                inside = 0 <= (x0 + x1) / 2 < 960 and 0 <= (y0 + y1) / 2 < 540 and (x1 - x0) >= 2 and (y1 - y0) >= 2
                rec[f"view{v:g}"] = {"in_view": bool(inside), "box_xyxy": [round(float(q), 2) for q in (x0, y0, x1, y1)],
                                     "fx": round(float(P[0, 0]), 4), "cx": 480.0, "cy": 270.0}
            out.append(rec)
    return out


def main():
    (REL / "zod_animal_range").mkdir(parents=True, exist_ok=True)
    (REL / "synthetic").mkdir(exist_ok=True)
    (REL / "tools").mkdir(exist_ok=True)
    ann = zod_annotations()
    (REL / "zod_animal_range/annotations.json").write_text(json.dumps(ann, indent=1))
    keys = ["frame_id", "annotation_uuid", "review_tag", "species", "time_of_day", "road_type", "depth_cam_m"]
    with (REL / "zod_animal_range/annotations_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(keys + ["in_view54.4", "in_view70"])
        for r in ann:
            w.writerow([r[k] for k in keys] + [r["view54.4"]["in_view"], r["view70"]["in_view"]])
    shutil.copy(ROOT / "output/synthetic_focal/focal_px.json", REL / "synthetic/focal_px.json")
    shutil.copy(ROOT / "output/synthetic_focal/report.json", REL / "synthetic/focal_recovery_report.json")
    folds = ["crossing8", "crossing9a", "crossing9b", "crossing9c", "crossing10", "crossing11", "crossing12",
             "crossing13", "crossing14", "crossing15"]
    (REL / "synthetic/loro_splits.json").write_text(json.dumps(
        {f"holdout_{s}": {"train": [t for t in folds if t != s], "test": [s]} for s in folds}, indent=1))
    for f in ("benchmark_make_views.py", "benchmark_evaluate.py"):
        shutil.copy(ROOT / "scripts" / f, REL / "tools" / f.replace("benchmark_", ""))
    n_tag = {}
    for r in ann:
        n_tag[r["review_tag"]] = n_tag.get(r["review_tag"], 0) + 1
    print(len(ann), "ZOD animals with LiDAR boxes;", n_tag)


if __name__ == "__main__":
    main()
