#!/usr/bin/env python3
"""Evaluate per-animal range predictions on the ZOD animal-range benchmark.

Predictions CSV columns: frame_id, annotation_uuid, view (54.4 or 70), pred_m.
Reports, per subset (hoofed / reindeer-and-deer / dogs / all reviewed non-statue / all in view):
  median absolute relative error with a frame-clustered bootstrap 95% interval,
  share within 10% of LiDAR, and the scaling slope (log predicted on log true range; 1 = correct scaling).
If both views are present, also reports the zoom test: median ratio of predictions 70/54.4 for the same animals
(a camera-aware estimate gives ~1; reading apparent size without the focal length gives ~1.36).
Usage: python evaluate.py --ann annotations.json --pred predictions.csv
"""
import argparse
import csv
import json

import numpy as np


def stats(z, p, frames, n_boot=2000, seed=0):
    rel = np.abs(p - z) / z
    out = {"n": int(len(z)), "frames": int(len(set(frames))), "median_rel": float(np.median(rel)),
           "within_10pct": float((rel <= 0.10).mean()), "slope": float(np.polyfit(np.log(z), np.log(p), 1)[0])}
    rng = np.random.default_rng(seed); uniq = np.array(sorted(set(frames))); fr = np.array(frames)
    idx = {f: np.flatnonzero(fr == f) for f in uniq}
    boots = [np.median(rel[np.concatenate([idx[f] for f in rng.choice(uniq, len(uniq))])]) for _ in range(n_boot)]
    out["median_rel_ci95"] = [float(q) for q in np.percentile(boots, [2.5, 97.5])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True); ap.add_argument("--pred", required=True)
    a = ap.parse_args()
    ann = {(r["frame_id"], r["annotation_uuid"]): r for r in json.load(open(a.ann))}
    preds = {}
    for r in csv.DictReader(open(a.pred)):
        preds[(r["frame_id"], r["annotation_uuid"], f"{float(r['view']):g}")] = float(r["pred_m"])
    subsets = {"hoofed": lambda r: r["review_tag"] == "ungulate",
               "reindeer_and_deer": lambda r: r["species"] in ("reindeer", "deer_or_reindeer"),
               "dogs": lambda r: r["review_tag"] == "dog",
               "all_in_view": lambda r: r["review_tag"] != "statue"}
    report = {}
    for view in ("54.4", "70"):
        for name, keep in subsets.items():
            rows = [(r["depth_cam_m"], preds[(k[0], k[1], view)], k[0]) for k, r in ann.items()
                    if keep(r) and r[f"view{view}"]["in_view"] and (k[0], k[1], view) in preds]
            if len(rows) >= 3:
                z, p, f = map(np.array, zip(*rows))
                report[f"{name}@{view}"] = stats(z.astype(float), p.astype(float), list(f))
    both = [(preds[(k[0], k[1], "70")], preds[(k[0], k[1], "54.4")]) for k in ann
            if (k[0], k[1], "70") in preds and (k[0], k[1], "54.4") in preds]
    if both:
        r = np.array(both)
        report["zoom_test_median_ratio_70_over_54"] = float(np.median(r[:, 0] / r[:, 1]))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
