#!/usr/bin/env python3
"""Compare range methods on the ten held-out rural roads with identical detections.

Per fold: the fold's detector-range checkpoint detects deer on its held-out road (conf 0.25, IoU 0.7);
detections are matched to ground truth (IoU >= 0.5). For each matched detection, range comes from
  head : the learned range head sampled at the detected-box center (paper's method)
  G0   : f * 1.41 m / detected box height
  G1   : f * HeightNet(crop of detected box) / detected box height  (fold model, held-out road unseen)
Joint success = detected and |range error| <= 5 m, over all ground-truth deer in the band.
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
from geo_range import FOLDS, OUT, TAG2SCENE, HeightNet, crop_box, predict_height  # noqa: E402
from loso_eval import load_gt  # noqa: E402

BANDS = [("30-40", 30, 40), ("40-50", 40, 50), ("50-60", 50, 60)]
METHODS = ("head", "g0", "g1")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main() -> None:
    focal = json.loads((ROOT / "output/synthetic_focal/focal_px.json").read_text())
    per_fold = {}
    for tag in FOLDS:
        weights = ROOT / f"output/loso_asset/{tag}/weights/best.pt"
        dist_model = load_dist_model(torch.device(DEVICE), weights)
        detector = YOLO(str(weights))
        net = HeightNet().to(DEVICE); net.load_state_dict(torch.load(OUT / f"heightnet_{tag}.pt")); net.eval()
        h0 = float(np.median([r["H_g0"] for r in json.loads((OUT / "loso_rows.json").read_text()) if r["tag"] == tag]))
        recs = []  # one per GT deer
        for stem, items in sorted(load_gt(tag).items()):
            items = [g for g in items if g["distance_m"] > 0]
            f = focal.get(f"{TAG2SCENE[tag]}/{int(stem.split('_')[1])}")
            if not items or f is None:
                continue
            img_path = ROOT / "data/loso/scenes" / tag / "images" / f"{stem}.jpg"
            img = cv2.imread(str(img_path)); h, w = img.shape[:2]
            lb = letterbox_map(img, 640, cv2.INTER_LINEAR)
            t = torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(DEVICE)
            dist_model(t); dmap = dist_model.dist_head(dist_model._feat["p3"])
            res = detector(str(img_path), device=DEVICE, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
            pb = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
            cf = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
            gt_boxes = np.array([g["box"] for g in items], np.float32)
            used, match = set(), {}
            for pi in np.argsort(-cf):
                ious = iou_one_to_many(pb[pi].astype(np.float32), gt_boxes)
                for gi in used:
                    ious[gi] = -1
                gi = int(np.argmax(ious))
                if ious[gi] >= 0.5:
                    used.add(gi); match[gi] = int(pi)
            det_boxes = [pb[match[gi]] for gi in range(len(items)) if gi in match]
            g1_h = predict_height(net, [crop_box(img, b) for b in det_boxes],
                                  [(b[2] - b[0]) / max(b[3] - b[1], 1e-3) for b in det_boxes]) if det_boxes else []
            k = 0
            for gi, g in enumerate(items):
                rec = {"z": g["distance_m"], "det": gi in match}
                if gi in match:
                    b = pb[match[gi]]; bh = max(b[3] - b[1], 1.0)
                    gx, gy = original_center_to_grid((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, w, h)
                    grid = torch.tensor([[[[gx, gy]]]], dtype=dmap.dtype, device=DEVICE)
                    rec["head"] = float(np.exp(F.grid_sample(dmap, grid, align_corners=False).item()))
                    rec["g0"] = f * h0 / bh
                    rec["g1"] = f * float(g1_h[k]) / bh
                    k += 1
                recs.append(rec)
        fold = {}
        for band, lo, hi in BANDS:
            b = [r for r in recs if lo <= r["z"] < hi]
            m = [r for r in b if r["det"]]
            entry = {"n": len(b), "recall": len(m) / len(b) if b else float("nan")}
            for meth in METHODS:
                err = np.array([r[meth] - r["z"] for r in m])
                entry[meth] = {"mae": float(np.abs(err).mean()), "bias": float(err.mean()),
                               "joint": float((np.abs(err) <= 5).sum() / len(b))}
            fold[band] = entry
        per_fold[tag] = fold
        print(tag, {bd: {mm: round(fold[bd][mm]["joint"], 3) for mm in METHODS} for bd, _, _ in BANDS}, flush=True)
    summary = {}
    for band, _, _ in BANDS:
        summary[band] = {"recall": float(np.mean([per_fold[t][band]["recall"] for t in FOLDS]))}
        for meth in METHODS:
            j = [per_fold[t][band][meth]["joint"] for t in FOLDS]
            summary[band][meth] = {"mae": float(np.mean([per_fold[t][band][meth]["mae"] for t in FOLDS])),
                                   "joint": float(np.mean(j)), "roads_pass_0.60": int(sum(x >= 0.60 for x in j))}
    (OUT / "loso_detected_comparison.json").write_text(json.dumps({"per_fold": per_fold, "mean": summary}, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
