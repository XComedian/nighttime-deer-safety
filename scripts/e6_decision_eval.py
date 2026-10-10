#!/usr/bin/env python3
"""E6: per-frame warning-decision evaluation with calibrated range uncertainty and false positives.

The rendered frames of a scene are independent draws (new deer distance and camera per frame), so they are
evaluated as independent decisions rather than replayed as a time series. For each of the ten rural
leave-one-scene-out folds, the held-out road's checkpoint is run on that road, keeping ALL detections
(conf >= 0.25), including unmatched false positives and detections on deer-free frames.

Decision (stationary deer, unconditional path conflict, base-case SSD with 2.5 s + 0.30 s and f = 0.35):
  need  = some ground-truth deer has true range < SSD
  alert = some detection has conservative range d_lo < SSD,   d_lo = d_hat * exp(-q)
q is a one-sided conformal quantile of e = log(d_hat) - log(d_true) over matched detections pooled from the
OTHER nine roads (leave-one-road-out calibration), so the evaluated road never informs its own buffer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_synthetic_distance import iou_one_to_many, letterbox_map, load_dist_model, original_center_to_grid  # noqa: E402
from loso_eval import load_gt  # noqa: E402

OUT = ROOT / "output/e6_decision"
SCENES = ROOT / "data/loso/scenes"
FOLDS = ["c8", "c9a", "c9b", "c9c", "c10", "c11", "c12", "c13", "c14", "c15"]
NAMES = {"c8": "Rural Baseline", "c9a": "Rural Straight", "c9b": "Rural Right Curve", "c9c": "Rural S-Curve",
         "c10": "Rural Crest", "c11": "Rural Farmland", "c12": "Rural Autumn", "c13": "Rural Scrub",
         "c14": "Rural Conifer", "c15": "Rural Shrub"}
SPEEDS = (15, 20, 25)
GRADES = (("level", 0.0), ("down8", -0.08))
COVERAGES = (None, 0.80, 0.90, 0.95, "detonly")  # None = point estimate (q = 0); detonly = warn on any detection
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def ssd_m(v_mph: float, grade: float, t: float = 2.8, f: float = 0.35) -> float:
    return (1.47 * v_mph * t + v_mph ** 2 / (30.0 * (f + grade))) * 0.3048


@torch.no_grad()
def run_fold(tag: str) -> list[dict]:
    cache = OUT / f"raw_{tag}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    weights = ROOT / f"output/loso_asset/{tag}/weights/best.pt"
    dist_model = load_dist_model(torch.device(DEVICE), weights)
    detector = YOLO(str(weights))
    frames = []
    for stem, items in sorted(load_gt(tag).items()):
        img_path = SCENES / tag / "images" / f"{stem}.jpg"
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        lb = letterbox_map(image, 640, cv2.INTER_LINEAR)
        t = torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(DEVICE)
        dist_model(t)
        dmap = dist_model.dist_head(dist_model._feat["p3"])
        res = detector(str(img_path), device=DEVICE, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
        pb = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        cf = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
        gts = [g for g in items if g["distance_m"] > 0]
        gt_boxes = np.array([g["box"] for g in gts], np.float32)
        dets, used = [], set()
        for pi in np.argsort(-cf):
            x0, y0, x1, y1 = pb[pi]
            gx, gy = original_center_to_grid((x0 + x1) / 2, (y0 + y1) / 2, w, h)
            grid = torch.tensor([[[[gx, gy]]]], dtype=dmap.dtype, device=DEVICE)
            pred = float(np.exp(F.grid_sample(dmap, grid, align_corners=False).item()))
            match = None
            if len(gt_boxes):
                ious = iou_one_to_many(pb[pi].astype(np.float32), gt_boxes)
                for gi in used:
                    ious[gi] = -1
                gi = int(np.argmax(ious))
                if ious[gi] >= 0.5:
                    used.add(gi); match = gi
            dets.append({"conf": float(cf[pi]), "pred_m": pred, "match": match})
        frames.append({"frame": stem, "gt_m": [g["distance_m"] for g in gts], "dets": dets})
    cache.write_text(json.dumps(frames))
    return frames


def residuals(frames: list[dict]) -> np.ndarray:
    return np.array([np.log(d["pred_m"]) - np.log(f["gt_m"][d["match"]])
                     for f in frames for d in f["dets"] if d["match"] is not None])


def evaluate(frames: list[dict], q: float, v: float, grade: float) -> dict:
    s = ssd_m(v, grade)
    c = {"need": 0, "hit": 0, "miss": 0, "miss_undetected": 0, "no_need": 0, "false_alert": 0,
         "empty": 0, "nuisance": 0}
    for f in frames:
        alert = any(d["pred_m"] * np.exp(-q) < s for d in f["dets"])
        if not f["gt_m"]:
            c["empty"] += 1; c["nuisance"] += alert
            continue
        need = min(f["gt_m"]) < s
        if need:
            c["need"] += 1
            if alert:
                c["hit"] += 1
            else:
                c["miss"] += 1
                needed = {i for i, g in enumerate(f["gt_m"]) if g < s}
                c["miss_undetected"] += not any(d["match"] in needed for d in f["dets"])
        else:
            c["no_need"] += 1; c["false_alert"] += alert
    return c


def rates(c: dict) -> dict:
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return {"miss_rate": div(c["miss"], c["need"]),
            "miss_from_detection_share": div(c["miss_undetected"], c["miss"]),
            "false_alert_rate": div(c["false_alert"], c["no_need"]),
            "nuisance_rate_empty_frames": div(c["nuisance"], c["empty"]),
            "need_frames": c["need"], "no_need_frames": c["no_need"], "empty_frames": c["empty"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {tag: run_fold(tag) for tag in FOLDS}
    res = {tag: residuals(fr) for tag, fr in data.items()}
    result = {"per_road": {}, "mean": {}}
    for tag in FOLDS:
        others = np.concatenate([res[o] for o in FOLDS if o != tag])
        road = {"name": NAMES[tag], "matched": int(len(res[tag])),
                "empty_frames": sum(not f["gt_m"] for f in data[tag]),
                "fp_detections": sum(d["match"] is None for f in data[tag] for d in f["dets"])}
        for cov in COVERAGES:
            if cov == "detonly":
                q, key = 50.0, "detection_only"          # d_lo ~ 0: any detection alerts
            else:
                q = 0.0 if cov is None else float(np.quantile(others, cov))
                key = "point" if cov is None else f"conformal{int(cov * 100)}"
            entry = {"q_log": q, "buffer_at_40m_m": 40 * (1 - np.exp(-q)),
                     "coverage_on_road": float(np.mean(res[tag] <= q)) if isinstance(cov, float) else None}
            for v in SPEEDS:
                for gname, g in GRADES:
                    entry[f"{v}mph_{gname}"] = rates(evaluate(data[tag], q, v, g))
            road[key] = entry
        result["per_road"][tag] = road
    for key in ["point"] + [f"conformal{int(c * 100)}" for c in COVERAGES if isinstance(c, float)] + ["detection_only"]:
        m = {"coverage_on_road_mean": None}
        covs = [result["per_road"][t][key]["coverage_on_road"] for t in FOLDS]
        if covs[0] is not None:
            m["coverage_on_road_mean"] = float(np.mean(covs)); m["coverage_on_road_min"] = float(np.min(covs))
        m["buffer_at_40m_m_mean"] = float(np.mean([result["per_road"][t][key]["buffer_at_40m_m"] for t in FOLDS]))
        for v in SPEEDS:
            for gname, _ in GRADES:
                sc = f"{v}mph_{gname}"
                m[sc] = {r: float(np.nanmean([result["per_road"][t][key][sc][r] for t in FOLDS]))
                         for r in ("miss_rate", "miss_from_detection_share", "false_alert_rate",
                                   "nuisance_rate_empty_frames")}
        result["mean"][key] = m
    result["ssd_m"] = {f"{v}mph_{g}": ssd_m(v, gv) for v in SPEEDS for g, gv in GRADES}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["ssd_m"], indent=1))
    for key, m in result["mean"].items():
        print(key, "coverage", m["coverage_on_road_mean"], "buffer@40m %.2f" % m["buffer_at_40m_m_mean"])
        for v in SPEEDS:
            for gname, _ in GRADES:
                r = m[f"{v}mph_{gname}"]
                print(f"   {v}mph_{gname}: miss {r['miss_rate']:.3f} (det share {r['miss_from_detection_share']:.2f})"
                      f" false {r['false_alert_rate']:.3f} nuisance {r['nuisance_rate_empty_frames']:.3f}")


if __name__ == "__main__":
    main()
