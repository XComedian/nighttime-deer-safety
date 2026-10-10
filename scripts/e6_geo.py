#!/usr/bin/env python3
"""E6 rerun with the geometry-anchored range: per-frame warning decisions for head vs G1 on identical
detections (all detections incl. false positives), leave-one-road-out conformal buffers per method."""
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
from e6_decision_eval import COVERAGES, GRADES, SPEEDS, evaluate, rates  # noqa: E402
from eval_synthetic_distance import iou_one_to_many, letterbox_map, load_dist_model, original_center_to_grid  # noqa: E402
from geo_range import FOLDS, OUT, TAG2SCENE, HeightNet, crop_box, predict_height  # noqa: E402
from loso_eval import load_gt  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RAW = OUT / "e6_raw"


@torch.no_grad()
def run_fold(tag: str, focal: dict) -> list[dict]:
    cache = RAW / f"{tag}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    weights = ROOT / f"output/loso_asset/{tag}/weights/best.pt"
    dist_model = load_dist_model(torch.device(DEVICE), weights); detector = YOLO(str(weights))
    net = HeightNet().to(DEVICE); net.load_state_dict(torch.load(OUT / f"heightnet_{tag}.pt")); net.eval()
    frames = []
    for stem, items in sorted(load_gt(tag).items()):
        f = focal.get(f"{TAG2SCENE[tag]}/{int(stem.split('_')[1])}")
        if f is None:
            continue
        img_path = ROOT / "data/loso/scenes" / tag / "images" / f"{stem}.jpg"
        img = cv2.imread(str(img_path)); h, w = img.shape[:2]
        lb = letterbox_map(img, 640, cv2.INTER_LINEAR)
        t = torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(DEVICE)
        dist_model(t); dmap = dist_model.dist_head(dist_model._feat["p3"])
        res = detector(str(img_path), device=DEVICE, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
        pb = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        cf = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
        gts = [g for g in items if g["distance_m"] > 0]
        gt_boxes = np.array([g["box"] for g in gts], np.float32)
        hh = predict_height(net, [crop_box(img, b) for b in pb], [(b[2] - b[0]) / max(b[3] - b[1], 1e-3) for b in pb]) if len(pb) else []
        dets, used = [], set()
        for pi in np.argsort(-cf):
            b = pb[pi]
            gx, gy = original_center_to_grid((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, w, h)
            grid = torch.tensor([[[[gx, gy]]]], dtype=dmap.dtype, device=DEVICE)
            match = None
            if len(gt_boxes):
                ious = iou_one_to_many(b.astype(np.float32), gt_boxes)
                for gi in used:
                    ious[gi] = -1
                gi = int(np.argmax(ious))
                if ious[gi] >= 0.5:
                    used.add(gi); match = gi
            dets.append({"head": float(np.exp(F.grid_sample(dmap, grid, align_corners=False).item())),
                         "g1": f * float(hh[pi]) / max(b[3] - b[1], 1.0), "match": match})
        frames.append({"gt_m": [g["distance_m"] for g in gts], "dets": dets})
    cache.write_text(json.dumps(frames))
    return frames


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    focal = json.loads((ROOT / "output/synthetic_focal/focal_px.json").read_text())
    data = {t: run_fold(t, focal) for t in FOLDS}
    result = {}
    for meth in ("head", "g1"):
        view = {t: [{"gt_m": fr["gt_m"], "dets": [{"pred_m": d[meth], "match": d["match"]} for d in fr["dets"]]}
                    for fr in data[t]] for t in FOLDS}
        res = {t: np.array([np.log(d["pred_m"]) - np.log(fr["gt_m"][d["match"]]) for fr in view[t]
                            for d in fr["dets"] if d["match"] is not None]) for t in FOLDS}
        per = {}
        for cov in COVERAGES:
            key = "point" if cov is None else ("detection_only" if cov == "detonly" else f"conformal{int(cov * 100)}")
            vals = {}
            for t in FOLDS:
                others = np.concatenate([res[o] for o in FOLDS if o != t])
                q = 50.0 if cov == "detonly" else (0.0 if cov is None else float(np.quantile(others, cov)))
                v = {f"{s}mph_{g}": rates(evaluate(view[t], q, s, gv)) for s in SPEEDS for g, gv in GRADES}
                v["coverage"] = float(np.mean(res[t] <= q)) if isinstance(cov, float) else None
                vals[t] = v
            agg = {sc: {r: float(np.nanmean([vals[t][sc][r] for t in FOLDS]))
                        for r in ("miss_rate", "false_alert_rate", "nuisance_rate_empty_frames")}
                   for sc in vals[FOLDS[0]] if sc != "coverage"}
            covs = [vals[t]["coverage"] for t in FOLDS]
            agg["coverage_mean_min"] = [float(np.mean(covs)), float(np.min(covs))] if covs[0] is not None else None
            per[key] = agg
        result[meth] = per
    (OUT / "e6_geo_summary.json").write_text(json.dumps(result, indent=2))
    for meth, per in result.items():
        for key, a in per.items():
            print(f"{meth:4s} {key:15s} cov {a['coverage_mean_min']} | 20mph miss {a['20mph_level']['miss_rate']:.3f} "
                  f"unneeded {a['20mph_level']['false_alert_rate']:.3f} | 25mph miss {a['25mph_level']['miss_rate']:.3f} "
                  f"unneeded {a['25mph_level']['false_alert_rate']:.3f} | nuisance25 {a['25mph_level']['nuisance_rate_empty_frames']:.3f}")


if __name__ == "__main__":
    main()
