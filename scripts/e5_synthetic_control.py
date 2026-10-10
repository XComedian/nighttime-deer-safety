#!/usr/bin/env python3
"""E5a control: the size-vs-range slope test on held-out synthetic roads, where range is known to work.

Between-instance regression of log range on log box height (GT boxes), for GT distance and for the
held-out fold's predicted range. Deer scale and pose vary, so the GT slope is the test's own ceiling.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from eval_synthetic_distance import letterbox_map, load_dist_model, original_center_to_grid
from loso_eval import load_gt

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAGS = {"c8": "c8", "c9a": "c9a", "c12": "c12", "c14": "c14"}


@torch.no_grad()
def main() -> None:
    out = {}
    for fold, tag in TAGS.items():
        model = load_dist_model(DEVICE, ROOT / f"output/loso_asset/{fold}/weights/best.pt")
        gts = load_gt(tag)
        lh, lz_gt, lz_pred = [], [], []
        for stem, items in gts.items():
            items = [g for g in items if g["distance_m"] > 0]
            if not items:
                continue
            img = cv2.imread(str(ROOT / "data/loso/scenes" / tag / "images" / f"{stem}.jpg"))
            h, w = img.shape[:2]
            lb = letterbox_map(img, 640, cv2.INTER_LINEAR)
            t = torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(DEVICE)
            model(t)
            dmap = model.dist_head(model._feat["p3"])
            for g in items:
                x0, y0, x1, y1 = g["box"]
                gx, gy = original_center_to_grid((x0 + x1) / 2, (y0 + y1) / 2, w, h)
                grid = torch.tensor([[[[gx, gy]]]], dtype=dmap.dtype, device=DEVICE)
                lz_pred.append(F.grid_sample(dmap, grid, align_corners=False).item())
                lz_gt.append(np.log(g["distance_m"])); lh.append(np.log(max(y1 - y0, 1.0)))
        lh, lz_gt, lz_pred = map(np.array, (lh, lz_gt, lz_pred))
        out[tag] = {"n": int(len(lh)), "slope_gt": float(np.polyfit(lh, lz_gt, 1)[0]),
                    "slope_pred": float(np.polyfit(lh, lz_pred, 1)[0]),
                    "r_gt": float(np.corrcoef(lh, lz_gt)[0, 1]), "r_pred": float(np.corrcoef(lh, lz_pred)[0, 1]),
                    # elasticity d log(pred) / d log(true): 1 = correct relative ranging, 0 = ignores distance
                    "elasticity_pred_vs_gt": float(np.polyfit(lz_gt, lz_pred, 1)[0])}
        print(tag, out[tag])
    p = ROOT / "output/e5_real_consistency/synthetic_control.json"
    p.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
