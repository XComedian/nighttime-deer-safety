#!/usr/bin/env python3
"""E4: does predicted road grade change the stopping-distance screen?

Held-out nuScenes night logs. Per keyframe:
  reference  = mean grade of the path the vehicle actually travels over the next 0-60 m
               (vehicle pitch from localization at future keyframes, distance-weighted)
  imu_only   = current vehicle pitch (gravity-referenced attitude; assumes constant grade)
  imu+X      = current pitch + relative road tilt ahead from source X
               X = LiDAR plane (sensor reference), or the geometry model's predicted
               normals / predicted depth in a fixed lower-image region (camera-only).
Grade errors are propagated through the AASHTO SSD equation used in the paper.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from nuscenes_geometry_experiment import (
    ROOT,
    NuScenesData,
    fit_road_plane,
    letterbox_map,
    load_geo_model,
    load_head_checkpoint,
    quat_matrix,
    read_manifest,
    sample_map,
)

OUT_DEFAULT = ROOT / "output/e4_grade_ssd"
GEO_OUT = ROOT / "output/nuscenes_geometry_full"
PATH_M = 60.0          # stopping path length used for the reference grade
MIN_COVER_M = 50.0     # require future trajectory to cover at least this distance
SPEEDS = (20, 25, 30)
T_TOTAL, F_BASE = 2.8, 0.35


def ssd_m(v_mph: float, grade: float, t: float = T_TOTAL, f: float = F_BASE) -> float:
    return (1.47 * v_mph * t + v_mph ** 2 / (30.0 * (f + grade))) * 0.3048


def boundary_mph(r_m: float, grade: float) -> float:
    lo, hi = 1.0, 80.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ssd_m(mid, grade) <= r_m else (lo, mid)
    return lo


def pose_grade_pct(rotation: list[float]) -> float:
    fwd = quat_matrix(rotation) @ np.array([1.0, 0.0, 0.0])
    return float(100.0 * fwd[2] / np.hypot(fwd[0], fwd[1]))


def ego_tilt_pct(normal_ego: np.ndarray) -> float:
    # slope of the plane along ego forward; invariant to normal sign
    return float(-100.0 * normal_ego[0] / normal_ego[2])


def roi_pixels(width: int, height: int, step: int = 8) -> np.ndarray:
    us, vs = np.meshgrid(np.arange(int(0.12 * width), int(0.88 * width), step),
                         np.arange(int(0.55 * height), height - 1, step))
    return np.stack([us.ravel(), vs.ravel()], 1).astype(np.float32)


@torch.no_grad()
def camera_tilts(model, image: np.ndarray, cam_cal: dict, device: torch.device, seed: int) -> tuple[float, float]:
    h, w = image.shape[:2]
    lb = letterbox_map(image, 640, cv2.INTER_LINEAR)
    tensor = torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(device)
    model(tensor)
    feat = model._f["p3"]
    scale = min(640 / h, 640 / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    left, top = (640 - nw) // 2, (640 - nh) // 2
    uv = roi_pixels(w, h)
    uv_lb = uv * scale + np.array([left, top], dtype=np.float32)
    r_cam = quat_matrix(cam_cal["rotation"])

    # normals route: mean predicted normal over the road ROI, rotated into the ego frame
    normals = F.normalize(sample_map(F.normalize(model.normal_head(feat), dim=1), uv_lb, device), dim=1)
    n_cam = normals.mean(0).cpu().numpy().astype(np.float64)
    n_ego = r_cam @ (n_cam / np.linalg.norm(n_cam))
    tilt_normal = ego_tilt_pct(n_ego)

    # depth route: back-project predicted depth, fit a road plane exactly as for the LiDAR reference
    depth = sample_map(model.depth_head(feat), uv_lb, device).flatten().clamp(0.1, 80).cpu().numpy()
    k_inv = np.linalg.inv(np.asarray(cam_cal["camera_intrinsic"], dtype=np.float64))
    rays = np.c_[uv, np.ones(len(uv))] @ k_inv.T
    xyz_cam = rays * depth[:, None]
    xyz_ego = xyz_cam @ r_cam.T + np.asarray(cam_cal["translation"])
    keep = (xyz_ego[:, 0] > 3.0) & (depth < 60.0)
    fit = fit_road_plane(xyz_ego[keep], seed=seed)
    tilt_depth = ego_tilt_pct(fit[0]) if fit is not None else float("nan")
    return tilt_normal, tilt_depth


def path_reference(s: np.ndarray, g: np.ndarray, k: int) -> float:
    ahead = s - s[k]
    if ahead[-1] < MIN_COVER_M:
        return float("nan")
    grid = np.linspace(0.0, min(PATH_M, ahead[-1]), 61)
    return float(np.interp(grid, ahead[k:], g[k:]).mean())


def summarize(rows: list[dict], estimators: list[str]) -> dict:
    out = {}
    for est in estimators:
        err = np.array([r[f"g_{est}"] - r["g_ref"] for r in rows])
        err = err[np.isfinite(err)]
        entry = {"n": int(len(err)), "grade_mae_pct": float(np.abs(err).mean()),
                 "grade_rmse_pct": float(np.sqrt((err ** 2).mean())), "grade_bias_pct": float(err.mean()),
                 "grade_p95_abs_pct": float(np.percentile(np.abs(err), 95))}
        for v in SPEEDS:
            d = np.array([ssd_m(v, r[f"g_{est}"] / 100) - ssd_m(v, r["g_ref"] / 100) for r in rows
                          if np.isfinite(r[f"g_{est}"])])
            entry[f"ssd_mae_m_{v}mph"] = float(np.abs(d).mean())
            entry[f"ssd_p95_abs_m_{v}mph"] = float(np.percentile(np.abs(d), 95))
        dv = np.array([boundary_mph(60, r[f"g_{est}"] / 100) - boundary_mph(60, r["g_ref"] / 100) for r in rows
                       if np.isfinite(r[f"g_{est}"])])
        entry["boundary60_mae_mph"] = float(np.abs(dv).mean())
        entry["boundary60_p95_abs_mph"] = float(np.percentile(np.abs(dv), 95))
        out[est] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    synthetic = load_geo_model(device).eval()
    adapted = load_geo_model(device).eval()
    load_head_checkpoint(adapted, GEO_OUT / "adapted/best_heads.pt")

    manifest = {split: read_manifest(GEO_OUT, split) for split in args.splits}
    scenes = {r["scene"] for rows in manifest.values() for r in rows}
    nusc = NuScenesData(ROOT / "data/nuscenes", "v1.0-trainval", scenes)

    rows = []
    for split, mrows in manifest.items():
        by_scene: dict[str, list[dict]] = {}
        for r in mrows:
            by_scene.setdefault(r["scene"], []).append(r)
        for scene, srows in sorted(by_scene.items()):
            samples = {s["token"]: s for s in nusc.scene_samples(scene)}
            srows.sort(key=lambda r: int(r["frame"]))
            recs = [nusc.sensor_record(samples[r["sample_token"]], "CAM_FRONT") for r in srows]
            xy = np.array([pose["translation"][:2] for _, _, pose in recs])
            s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            g = np.array([pose_grade_pct(pose["rotation"]) for _, _, pose in recs])
            for k, (r, (cam_sd, cam_cal, _)) in enumerate(zip(srows, recs)):
                g_ref = path_reference(s, g, k)
                if not np.isfinite(g_ref):
                    continue
                image = cv2.imread(str(ROOT / r["image"]))
                seed = int(r["frame"]) + 1000 * int(scene.split("-")[-1])
                syn_n, syn_d = camera_tilts(synthetic, image, cam_cal, device, seed)
                ada_n, ada_d = camera_tilts(adapted, image, cam_cal, device, seed)
                lidar_rel = float(r["grade_pct"]) if r["grade_pct"] not in ("", "nan") else float("nan")
                rows.append({
                    "split": split, "scene": scene, "frame": int(r["frame"]),
                    "g_ref": g_ref, "g_imu_only": g[k],
                    "rel_lidar": lidar_rel, "rel_adapted_normal": ada_n, "rel_adapted_depth": ada_d,
                    "rel_synthetic_normal": syn_n, "rel_synthetic_depth": syn_d,
                    "g_imu_lidar": g[k] + lidar_rel,
                    "g_imu_adapted_normal": g[k] + ada_n, "g_imu_adapted_depth": g[k] + ada_d,
                    "g_imu_synthetic_normal": g[k] + syn_n, "g_imu_synthetic_depth": g[k] + syn_d,
                })
            print(f"{split} {scene}: {sum(x['scene'] == scene for x in rows)} frames with >= {MIN_COVER_M} m ahead")

    with (args.out / "frame_grades.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    estimators = ["imu_only", "imu_lidar", "imu_adapted_normal", "imu_adapted_depth",
                  "imu_synthetic_normal", "imu_synthetic_depth"]
    result = {"assumptions": {"path_m": PATH_M, "min_cover_m": MIN_COVER_M, "t_total_s": T_TOTAL,
                              "friction": F_BASE, "speeds_mph": SPEEDS}}
    for split in args.splits:
        sel = [r for r in rows if r["split"] == split]
        result[split] = summarize(sel, estimators)
        # relative tilt agreement with the LiDAR plane (camera vs sensor, no trajectory involved)
        rel = {}
        for src in ["adapted_normal", "adapted_depth", "synthetic_normal", "synthetic_depth"]:
            e = np.array([x[f"rel_{src}"] - x["rel_lidar"] for x in sel])
            e = e[np.isfinite(e)]
            rel[src] = {"n": int(len(e)), "mae_pct": float(np.abs(e).mean()), "bias_pct": float(e.mean())}
        result[split]["relative_tilt_vs_lidar"] = rel
        result[split]["reference_grade_pct"] = {
            "mean": float(np.mean([x["g_ref"] for x in sel])),
            "p5_p95": [float(q) for q in np.percentile([x["g_ref"] for x in sel], [5, 95])]}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
