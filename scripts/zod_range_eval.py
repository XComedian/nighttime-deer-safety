#!/usr/bin/env python3
"""E5b: real metric range on ZOD animals (LiDAR 3D boxes) for the synthetic-trained detector-range model.

Each ZOD front image (3848x2168 Kannala-Brandt fisheye, 120 deg HFOV) is resampled into a virtual
960x540 pinhole view with the same optical center and axis, matching the synthetic training camera
(24-50 mm on a 36 mm sensor = 40-74 deg HFOV). Animal 3D boxes (LiDAR frame) are moved to the camera
frame; ground truth range = camera-frame forward depth Z of the box center (synthetic labels are
z-depth). 2D boxes are undistorted into the virtual view.

Outputs per animal: range-head prediction at the GT box center, detection match, and three geometric
baselines: flat-ground (camera height + box bottom), size-oracle (true LiDAR height / box height), and
size-prior (H = 1.0 m / box height).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_synthetic_distance import iou_one_to_many, letterbox_map, load_dist_model, original_center_to_grid  # noqa: E402

ZOD = Path(os.environ.get("ZOD_ROOT", ROOT / "data/zod")) / "single_frames"
OUT = ROOT / "output/zod"
W, H = 960, 540
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# Camera height above the road: median of (camera-frame y of box bottom) over 1,481 annotated animals and
# vehicles 5-40 m ahead; the extrinsics height (1.13 m) is relative to the ego origin, not the ground.
CAM_HEIGHT_M = 1.53


def quat_to_rot(qw, qx, qy, qz):
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def virtual_camera(calib: dict, hfov_deg: float):
    fc = calib["FC"]
    K = np.array(fc["intrinsics"])[:, :3]
    D = np.array(fc["distortion"], dtype=np.float64)
    f = (W / 2) / np.tan(np.radians(hfov_deg / 2))
    P = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, (W, H), cv2.CV_32FC1)
    return K, D, P, m1, m2


def lidar_to_cam(calib: dict) -> np.ndarray:
    cam = np.array(calib["FC"]["extrinsics"]); lid = np.array(calib["FC"]["lidar_extrinsics"])
    return np.linalg.inv(cam) @ lid          # both are poses in the ego frame


def animals(fid: str, calib: dict, K, D, P) -> list[dict]:
    T = lidar_to_cam(calib)
    cam_h = CAM_HEIGHT_M
    out = []
    for a in json.loads((ZOD / fid / "annotations/object_detection.json").read_text()):
        pr = a["properties"]
        if pr.get("class") != "Animal" or "location_3d" not in pr:
            continue
        c = np.r_[pr["location_3d"]["coordinates"], 1.0]
        z = float((T @ c)[2])
        if z < 3:
            continue
        pts = np.array(a["geometry"]["coordinates"], dtype=np.float64).reshape(-1, 1, 2)
        und = cv2.fisheye.undistortPoints(pts, K, D, R=np.eye(3), P=P).reshape(-1, 2)
        x0, y0 = und.min(0); x1, y1 = und.max(0)
        if x1 < 0 or y1 < 0 or x0 > W or y0 > H or (x1 - x0) < 2 or (y1 - y0) < 2:
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        out.append({"frame": fid, "gt_z": z, "box": [float(x0), float(y0), float(x1), float(y1)],
                    "len": pr["size_3d_length"], "hgt": pr["size_3d_height"], "cam_h": cam_h,
                    "occlusion": pr.get("occlusion_ratio"), "unclear": pr.get("unclear"),
                    "rel_pos": pr.get("relative_position")})
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hfov", type=float, default=54.4)
    ap.add_argument("--fold", default="c8")
    ap.add_argument("--save-views", action="store_true")
    args = ap.parse_args()
    weights = ROOT / f"output/loso_asset/{args.fold}/weights/best.pt"
    dist_model = load_dist_model(torch.device(DEVICE), weights)
    detector = YOLO(str(weights))
    animal_frames = sorted({r["frame"] for r in csv.DictReader((OUT / "animal_annotations.csv").open())})
    negatives = json.loads((OUT / "negative_frames.json").read_text())
    tag = f"{args.fold}_hfov{args.hfov:g}"
    view_dir = OUT / "views" / f"hfov{args.hfov:g}"
    if args.save_views:
        view_dir.mkdir(parents=True, exist_ok=True)
    rows, neg_rows = [], []
    for fid in animal_frames + negatives:
        info = json.loads((ZOD / fid / "info.json").read_text())
        img_path = ZOD.parent / info["camera_frames"]["front_blur"][0]["filepath"] if "camera_frames" in info \
            else ZOD.parent / info["front_blur"][0]["filepath"]
        if not img_path.exists():
            continue
        calib = json.loads((ZOD / fid / "calibration.json").read_text())
        meta = json.loads((ZOD / fid / "metadata.json").read_text())
        K, D, P, m1, m2 = virtual_camera(calib, args.hfov)
        view = cv2.remap(cv2.imread(str(img_path)), m1, m2, cv2.INTER_LINEAR)
        if args.save_views:
            cv2.imwrite(str(view_dir / f"{fid}.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lb = letterbox_map(view, 640, cv2.INTER_LINEAR)
        t = torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(DEVICE)
        dist_model(t)
        dmap = dist_model.dist_head(dist_model._feat["p3"])
        res = detector(view, device=DEVICE, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
        pb = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        cf = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)

        def sample(x, y):
            gx, gy = original_center_to_grid(x, y, W, H)
            g = torch.tensor([[[[gx, gy]]]], dtype=dmap.dtype, device=DEVICE)
            return float(np.exp(F.grid_sample(dmap, g, align_corners=False).item()))

        if fid in negatives:
            neg_rows.append({"frame": fid, "time_of_day": meta["time_of_day"], "n_det": int(len(pb)),
                             "max_conf": float(cf.max()) if len(cf) else 0.0})
            continue
        anns = animals(fid, calib, K, D, P)
        gt_boxes = np.array([a["box"] for a in anns], np.float32)
        used, match = set(), {}
        for pi in np.argsort(-cf):
            if not len(gt_boxes):
                break
            ious = iou_one_to_many(pb[pi].astype(np.float32), gt_boxes)
            for gi in used:
                ious[gi] = -1
            gi = int(np.argmax(ious))
            if ious[gi] >= 0.5:
                used.add(gi); match[gi] = int(pi)
        f = P[0, 0]
        for gi, a in enumerate(anns):
            x0, y0, x1, y1 = a["box"]
            bh = y1 - y0
            det = gi in match
            pred_det = sample((pb[match[gi]][0] + pb[match[gi]][2]) / 2, (pb[match[gi]][1] + pb[match[gi]][3]) / 2) if det else None
            horizon = H / 2                              # level virtual camera on the optical axis
            rows.append({**{k: a[k] for k in ("frame", "gt_z", "len", "hgt", "occlusion", "unclear", "rel_pos")},
                         "time_of_day": meta["time_of_day"], "road_type": meta["road_type"],
                         "box_h": bh, "box_w": x1 - x0,
                         "pred_gtcenter": sample((x0 + x1) / 2, (y0 + y1) / 2),
                         "detected": int(det), "det_conf": float(cf[match[gi]]) if det else None,
                         "pred_detbox": pred_det,
                         "base_flatground": (f * a["cam_h"] / (y1 - horizon)) if y1 > horizon + 1 else None,
                         "base_size_oracle": f * a["hgt"] / bh,
                         "base_size_prior1m": f * 1.0 / bh})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"range_rows_{tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (OUT / f"negatives_{tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(neg_rows[0])); w.writeheader(); w.writerows(neg_rows)
    print(f"{tag}: {len(rows)} animals in view, {len(neg_rows)} negative frames")


if __name__ == "__main__":
    main()
