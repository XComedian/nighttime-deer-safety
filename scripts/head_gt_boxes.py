#!/usr/bin/env python3
"""Learned range head sampled at ground-truth box centers on each held-out rural road (fold checkpoint)."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from eval_synthetic_distance import letterbox_map, load_dist_model, original_center_to_grid
from geo_range import FOLDS
from loso_eval import load_gt

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main():
    rows = []
    for tag in FOLDS:
        m = load_dist_model(torch.device("cuda"), ROOT / f"output/loso_asset/{tag}/weights/best.pt")
        for stem, items in sorted(load_gt(tag).items()):
            items = [g for g in items if g["distance_m"] > 0]
            if not items:
                continue
            img = cv2.imread(str(ROOT / "data/loso/scenes" / tag / "images" / f"{stem}.jpg")); h, w = img.shape[:2]
            t = torch.from_numpy(cv2.cvtColor(letterbox_map(img, 640, cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].cuda()
            m(t); dmap = m.dist_head(m._feat["p3"])
            for g in items:
                x0, y0, x1, y1 = g["box"]
                gx, gy = original_center_to_grid((x0 + x1) / 2, (y0 + y1) / 2, w, h)
                p = float(np.exp(F.grid_sample(dmap, torch.tensor([[[[gx, gy]]]], dtype=dmap.dtype, device="cuda"), align_corners=False).item()))
                rows.append({"tag": tag, "gt_z": g["distance_m"], "pred": p})
    (ROOT / "output/depth_baselines/head_gtbox_rendered.json").write_text(json.dumps(rows))
    print(len(rows))


if __name__ == "__main__":
    main()
