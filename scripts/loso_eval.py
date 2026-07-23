#!/usr/bin/env python3
"""Evaluate LOSO folds: each fold's checkpoint on its held-out scene, then aggregate.

For every fold <held>, load output/loso/<held>/weights/best.pt, run the detector on the
held-out scene at conf 0.25 / IoU 0.7, match detections to GT at IoU 0.5, sample the
distance head at each matched detected-box center, and compute per-band (30-40/40-50/
50-60 m) recall, matched MAE, and joint (detected AND |err|<=5 m). The detection-only
envelope per fold is the contiguous run of bands with recall >= 0.60. Aggregate across
folds as mean +/- std -> a distribution of independent-scene generalization, which is
the leave-one-scene-out answer to "why this split".

Usage: loso_eval.py c7 c8 c9a c9b c9c c10
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
from eval_synthetic_distance import (ARCH, iou_one_to_many, letterbox_map,  # noqa: E402
                                     load_dist_model, original_center_to_grid)

SCENES = ROOT / "data" / "loso" / "scenes"
# Checkpoint/report dir; override with LOSO_OUT so separate experiments (e.g. the
# appearance-generalization run) do not read each other's checkpoints.
import os  # noqa: E402
LOSO_OUT = Path(os.environ.get("LOSO_OUT", ROOT / "output" / "loso"))
# Allow a reproducibility recheck to read frozen checkpoints from one directory
# while writing reports to another, so the original report is never overwritten.
LOSO_WEIGHTS = Path(os.environ.get("LOSO_WEIGHTS", LOSO_OUT))
# Optional one-checkpoint override for a controlled cross-dataset re-evaluation.
LOSO_SINGLE_WEIGHT = os.environ.get("LOSO_SINGLE_WEIGHT")
BANDS = [("30-40", 30, 40), ("40-50", 40, 50), ("50-60", 50, 60)]
RECALL_SCREEN = 0.60
TOL = 5.0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_gt(tag: str) -> dict[str, list[dict]]:
    scene = SCENES / tag
    dists = json.loads((scene / "distances.json").read_text())
    gts: dict[str, list[dict]] = {}
    for img in sorted((scene / "images").glob("*.jpg")):
        stem = img.stem
        im = cv2.imread(str(img))
        h, w = im.shape[:2]
        lbl = (scene / "labels" / f"{stem}.txt").read_text().splitlines()
        dlist = dists.get(stem, [])
        items = []
        for i, line in enumerate(lbl):
            p = line.split()
            if len(p) != 5:
                continue
            cx, cy, bw, bh = (float(v) for v in p[1:])
            x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x1, y1 = (cx + bw / 2) * w, (cy + bh / 2) * h
            dist = dlist[i][2] if i < len(dlist) else -1.0
            items.append({"box": [x0, y0, x1, y1], "distance_m": dist, "wh": (w, h)})
        gts[stem] = items
    return gts


@torch.no_grad()
def eval_fold(tag: str, weights: Path) -> dict:
    gts = load_gt(tag)
    dist_model = load_dist_model(torch.device(DEVICE), weights)
    detector = YOLO(str(weights))
    per = {b: {"gt": 0, "det": 0, "abserr": [], "joint": 0} for b, _, _ in BANDS}

    for img in sorted((SCENES / tag / "images").glob("*.jpg")):
        items = gts[img.stem]
        image = cv2.imread(str(img))
        h, w = image.shape[:2]
        lb = letterbox_map(image, 640, cv2.INTER_LINEAR)
        t = (torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))
             .permute(2, 0, 1).float().div(255)[None].to(DEVICE))
        dist_model(t)
        dmap = dist_model.dist_head(dist_model._feat["p3"])

        res = detector(str(img), device=DEVICE, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
        pb = res.boxes.xyxy.cpu().numpy().astype(np.float32) if len(res.boxes) else np.zeros((0, 4), np.float32)
        cf = res.boxes.conf.cpu().numpy().astype(np.float32) if len(res.boxes) else np.zeros(0, np.float32)
        gt_boxes = np.array([g["box"] for g in items], np.float32)
        matched: dict[int, int] = {}
        used: set[int] = set()
        for pi in np.argsort(-cf):
            if not len(gt_boxes):
                break
            ious = iou_one_to_many(pb[pi], gt_boxes)
            for gi in used:
                ious[gi] = -1
            gi = int(np.argmax(ious))
            if ious[gi] >= 0.5:
                used.add(gi); matched[gi] = int(pi)

        for gi, g in enumerate(items):
            band = next((b for b, lo, hi in BANDS if lo <= g["distance_m"] < hi), None)
            if band is None:
                continue
            per[band]["gt"] += 1
            if gi in matched:
                per[band]["det"] += 1
                box = pb[matched[gi]]
                gx, gy = original_center_to_grid(float((box[0]+box[2])/2), float((box[1]+box[3])/2), w, h)
                grid = torch.tensor([[[[gx, gy]]]], device=DEVICE, dtype=dmap.dtype)
                pred = float(np.exp(F.grid_sample(dmap, grid, align_corners=False).item()))
                err = abs(pred - g["distance_m"])
                per[band]["abserr"].append(err)
                if err <= TOL:
                    per[band]["joint"] += 1

    out = {"held": tag, "bands": {}}
    envelope = 0
    contiguous = True
    for b, _, hi in BANDS:
        d = per[b]
        recall = d["det"] / d["gt"] if d["gt"] else float("nan")
        mae = float(np.mean(d["abserr"])) if d["abserr"] else float("nan")
        joint = d["joint"] / d["gt"] if d["gt"] else float("nan")
        out["bands"][b] = {"gt": d["gt"], "recall": recall, "mae": mae, "joint": joint}
        if contiguous and d["gt"] and recall >= RECALL_SCREEN:
            envelope = hi
        else:
            contiguous = False
    out["detection_envelope_m"] = envelope
    return out


def main() -> None:
    tags = sys.argv[1:]
    LOSO_OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for tag in tags:
        w = (Path(LOSO_SINGLE_WEIGHT) if LOSO_SINGLE_WEIGHT
             else LOSO_WEIGHTS / tag / "weights" / "best.pt")
        if not w.exists():
            print(f"SKIP {tag}: no checkpoint at {w}")
            continue
        r = eval_fold(tag, w)
        results.append(r)
        print(f"fold {tag}: envelope {r['detection_envelope_m']} m | "
              + " ".join(f"{b} R={r['bands'][b]['recall']:.3f}" for b, _, _ in BANDS))
    (LOSO_OUT / "loso_results.json").write_text(json.dumps(results, indent=2))

    def agg(key_fn):
        vals = [key_fn(r) for r in results if not np.isnan(key_fn(r))]
        return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (float("nan"), float("nan"), 0)

    lines = ["# Leave-One-Scene-Out Cross-Validation", "",
             f"Folds: {[r['held'] for r in results]}. Each row trains on the other scenes and",
             "tests zero-shot on the held-out scene (conf 0.25, IoU 0.5, joint tol 5 m).", "",
             "## Per-fold detection-only envelope",
             "", "| held-out scene | envelope (m) | R 30-40 | R 40-50 | R 50-60 |",
             "|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['held']} | {r['detection_envelope_m']} | "
                     + " | ".join(f"{r['bands'][b]['recall']:.3f}" for b, _, _ in BANDS) + " |")
    env_m, env_s, _ = agg(lambda r: r["detection_envelope_m"])
    lines += ["", f"**Detection-only envelope across folds: {env_m:.0f} +/- {env_s:.0f} m**", "",
              "## Per-band aggregates (mean +/- std across folds)", "",
              "| band | recall | matched MAE (m) | joint <=5 m |", "|---|---:|---:|---:|"]
    for b, _, _ in BANDS:
        rm, rs, _ = agg(lambda r: r["bands"][b]["recall"])
        mm, ms, _ = agg(lambda r: r["bands"][b]["mae"])
        jm, js, _ = agg(lambda r: r["bands"][b]["joint"])
        lines.append(f"| {b} | {rm:.3f} +/- {rs:.3f} | {mm:.2f} +/- {ms:.2f} | {jm:.3f} +/- {js:.3f} |")
    (LOSO_OUT / "loso_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-10:]))


if __name__ == "__main__":
    main()
