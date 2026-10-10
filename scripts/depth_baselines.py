#!/usr/bin/env python3
"""Zero-shot metric-depth baselines for per-animal range (run in .venv_depth).

Models: Metric3D v2 (canonical-camera transform; uses known focal length), UniDepth v2 (takes intrinsics),
Depth Anything V2 Metric-Outdoor (not camera-aware). Range for a box = median predicted depth over the box's
central region (middle 50% of width, middle 60% of height), compared with forward-depth ground truth.

Sets: ZOD virtual pinhole views (known focal), the ten held-out rendered rural roads (recovered focal),
and the dusk-clip deer tracks (focal unknown; scale-free slope only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/depth_baselines"
SCENES = ROOT / "data/loso/scenes"
FOLDS = ["c8", "c9a", "c9b", "c9c", "c10", "c11", "c12", "c13", "c14", "c15"]
DEV = "cuda"


class Metric3D:
    def __init__(self, size):
        self.m = torch.hub.load("yvanyin/metric3d", f"metric3d_vit_{size}", pretrain=True, trust_repo=True).to(DEV).eval()
        self.input_size = (616, 1064)

    @torch.no_grad()
    def __call__(self, bgr, fx, cx, cy):
        rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb0.shape[:2]
        s = min(self.input_size[0] / h, self.input_size[1] / w)
        rgb = cv2.resize(rgb0, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
        ph, pw = self.input_size[0] - rgb.shape[0], self.input_size[1] - rgb.shape[1]
        pad = [ph // 2, ph - ph // 2, pw // 2, pw - pw // 2]
        rgb = cv2.copyMakeBorder(rgb, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        x = ((torch.from_numpy(rgb.transpose(2, 0, 1)).float() - mean) / std)[None].to(DEV)
        depth, _, _ = self.m.inference({"input": x})
        depth = depth.squeeze()[pad[0]:depth.shape[-2] - pad[1], pad[2]:depth.shape[-1] - pad[3]]
        depth = F.interpolate(depth[None, None], (h, w), mode="bilinear").squeeze()
        return (depth * (fx * s / 1000.0)).cpu().numpy()          # canonical focal 1000 -> real camera


class UniDepth:
    def __init__(self, size):
        from unidepth.models import UniDepthV2
        self.m = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-vit{size}14").to(DEV).eval()

    @torch.no_grad()
    def __call__(self, bgr, fx, cx, cy):
        rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        if fx is None:
            pred = self.m.infer(rgb)
        else:
            K = torch.tensor([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=torch.float32)
            pred = self.m.infer(rgb, K)
        return pred["depth"].squeeze().cpu().numpy()


class DepthAnythingMetric:
    def __init__(self, size):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        name = f"depth-anything/Depth-Anything-V2-Metric-Outdoor-{size}-hf"
        self.p = AutoImageProcessor.from_pretrained(name)
        self.m = AutoModelForDepthEstimation.from_pretrained(name).to(DEV).eval()

    @torch.no_grad()
    def __call__(self, bgr, fx, cx, cy):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.p(images=rgb, return_tensors="pt").to(DEV)
        d = self.m(**inp).predicted_depth
        return F.interpolate(d[:, None], rgb.shape[:2], mode="bilinear").squeeze().cpu().numpy()


MODELS = {"metric3d_s": lambda: Metric3D("small"), "metric3d_l": lambda: Metric3D("large"),
          "unidepth_s": lambda: UniDepth("s"), "unidepth_l": lambda: UniDepth("l"),
          "dav2_metric_b": lambda: DepthAnythingMetric("Base")}


def box_depth(depth: np.ndarray, box) -> float:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    xa, xb = int(round(x0 + 0.25 * w)), int(round(x1 - 0.25 * w))
    ya, yb = int(round(y0 + 0.20 * h)), int(round(y1 - 0.20 * h))
    H, W = depth.shape
    xa, ya = max(0, min(xa, W - 1)), max(0, min(ya, H - 1))
    xb, yb = max(xa + 1, min(xb, W)), max(ya + 1, min(yb, H))
    return float(np.median(depth[ya:yb, xa:xb]))


def rendered_items(stride: int) -> list[dict]:
    focal = json.loads((ROOT / "output/synthetic_focal/focal_px.json").read_text())
    items = []
    for tag in FOLDS:
        d = SCENES / tag
        dists = json.loads((d / "distances.json").read_text())
        for i, img in enumerate(sorted((d / "images").glob("*.jpg"))):
            if i % stride:
                continue
            stem = img.stem
            f = focal.get(f"crossing{tag[1:]}/{int(stem.split('_')[1])}")
            lines = (d / "labels" / f"{stem}.txt").read_text().split()
            dl = dists.get(stem, [])
            boxes = []
            for k in range(0, len(lines), 5):
                j = k // 5
                if j >= len(dl) or dl[j][2] <= 0:
                    continue
                cx, cy, bw, bh = (float(v) for v in lines[k + 1:k + 5])
                boxes.append({"box": [(cx - bw / 2) * 960, (cy - bh / 2) * 540, (cx + bw / 2) * 960, (cy + bh / 2) * 540], "gt_z": dl[j][2]})
            if f is not None and boxes:
                items.append({"tag": tag, "image": str(img), "fx": f, "cx": 480.0, "cy": 270.0, "boxes": boxes})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--render-stride", type=int, default=1)
    args = ap.parse_args()
    model = MODELS[args.model]()
    ev = json.loads((OUT / "eval_boxes.json").read_text())
    res = {"model": args.model}
    for key in ("zod_54.4", "zod_70"):
        by_img: dict[str, list[dict]] = {}
        for r in ev[key]:
            by_img.setdefault(r["image"], []).append(r)
        rows = []
        for img, rs in by_img.items():
            depth = model(cv2.imread(img), rs[0]["fx"], rs[0]["cx"], rs[0]["cy"])
            for r in rs:
                rows.append({**{k: r[k] for k in ("frame", "gt_z", "tag", "species", "time_of_day")}, "pred": box_depth(depth, r["box"])})
        res[key] = rows
        print(key, len(rows), flush=True)
    rows = []
    for it in rendered_items(args.render_stride):
        depth = model(cv2.imread(it["image"]), it["fx"], it["cx"], it["cy"])
        for b in it["boxes"]:
            rows.append({"tag": it["tag"], "gt_z": b["gt_z"], "pred": box_depth(depth, b["box"])})
    res["rendered"] = rows
    print("rendered", len(rows), flush=True)
    tracks = []
    for tr in ev["dashcam"]:
        preds = []
        for d in tr:
            img = cv2.imread(d["image"])
            fx = None if args.model.startswith("unidepth") else (img.shape[1] / 2) / np.tan(np.radians(35))  # assumed 70 deg
            preds.append(box_depth(model(img, fx, img.shape[1] / 2, img.shape[0] / 2), d["box"]))
        tracks.append({"pred": preds, "h": [d["box"][3] - d["box"][1] for d in tr]})
    res["dashcam"] = tracks
    (OUT / f"{args.model}.json").write_text(json.dumps(res))
    print("done", args.model, flush=True)


if __name__ == "__main__":
    main()
