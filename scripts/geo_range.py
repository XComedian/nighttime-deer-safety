#!/usr/bin/env python3
"""Geometry-anchored deer range:  Z = f * H / h_box.

f      camera focal length in pixels (known from calibration; recovered per frame for the renders)
h_box  box height in pixels (ground-truth or detected box)
H      physical height (m) of the animal's box extent, estimated by
         G0: constant = median training H* (no learning), or
         G1: a small CNN on the box crop resized to 64x64 (scale- and camera-blind) -> log H.
Training target per rendered box: H* = Z * h_box / f.

Protocol: leave-one-road-out over the ten rural roads (train on nine, test the held-out road), plus one
model on all ten roads for transfer to ZOD and the real dashcam clips.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "data/loso/scenes"
OUT = ROOT / "output/geo_range"
FOLDS = ["c8", "c9a", "c9b", "c9c", "c10", "c11", "c12", "c13", "c14", "c15"]
TAG2SCENE = {t: "crossing" + t[1:] for t in FOLDS}
CROP = 64
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def crop_box(img: np.ndarray, box, ctx: float = 0.10) -> np.ndarray:
    """Square, letterboxed crop of a box with a little context, resized to CROP x CROP."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, x1 = x0 - ctx * w, x1 + ctx * w
    y0, y1 = y0 - ctx * h, y1 + ctx * h
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    M = np.array([[CROP / side, 0, CROP / 2 - cx * CROP / side], [0, CROP / side, CROP / 2 - cy * CROP / side]])
    return cv2.warpAffine(img, M, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def load_samples(tag: str, focal: dict) -> list[dict]:
    d = SCENES / tag
    dists = json.loads((d / "distances.json").read_text())
    out = []
    for img_path in sorted((d / "images").glob("*.jpg")):
        stem = img_path.stem
        f = focal.get(f"{TAG2SCENE[tag]}/{int(stem.split('_')[1])}")
        lines = (d / "labels" / f"{stem}.txt").read_text().split()
        if f is None or not lines:
            continue
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        dl = dists.get(stem, [])
        rows = [lines[i:i + 5] for i in range(0, len(lines), 5)]
        for i, r in enumerate(rows):
            if i >= len(dl) or dl[i][2] <= 0:
                continue
            cx, cy, bw, bh = (float(v) for v in r[1:])
            box = [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H]
            z = dl[i][2]
            h_px = box[3] - box[1]
            out.append({"tag": tag, "stem": stem, "box": box, "z": z, "f": f, "h_px": h_px,
                        "H": z * h_px / f, "aspect": (box[2] - box[0]) / max(h_px, 1e-3),
                        "crop": crop_box(img, box)})
    return out


class HeightNet(nn.Module):
    """Crop (scale-normalized) + box aspect ratio -> log physical height."""
    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.SiLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.SiLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 128))
        self.head = nn.Sequential(nn.Linear(128 + 1, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x, aspect):
        z = self.features(x).mean((2, 3))
        return self.head(torch.cat([z, torch.log(aspect)[:, None]], 1)).squeeze(1)


def to_tensor(crops: list[np.ndarray]) -> torch.Tensor:
    a = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(0, 3, 1, 2)


def train_height(samples: list[dict], epochs: int = 20, seed: int = 0) -> HeightNet:
    torch.manual_seed(seed); np.random.seed(seed)
    x = to_tensor([s["crop"] for s in samples])
    asp = torch.tensor([s["aspect"] for s in samples], dtype=torch.float32)
    y = torch.log(torch.tensor([s["H"] for s in samples], dtype=torch.float32))
    net = HeightNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((len(x) + 127) // 128))
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 128):
            idx = perm[i:i + 128]
            xb = x[idx].to(DEVICE)
            if np.random.rand() < 0.5:
                xb = xb.flip(3)                                          # left/right facing
            xb = (xb * torch.empty(len(xb), 1, 1, 1, device=DEVICE).uniform_(0.4, 1.6)).clamp(0, 1)  # exposure
            loss = F.smooth_l1_loss(net(xb, asp[idx].to(DEVICE)), y[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval()


@torch.no_grad()
def predict_height(net: HeightNet, crops: list[np.ndarray], aspects: list[float]) -> np.ndarray:
    out = []
    for i in range(0, len(crops), 512):
        xb = to_tensor(crops[i:i + 512]).to(DEVICE)
        ab = torch.tensor(aspects[i:i + 512], dtype=torch.float32, device=DEVICE)
        out.append(torch.exp(net(xb, ab)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    focal = json.loads((ROOT / "output/synthetic_focal/focal_px.json").read_text())
    data = {t: load_samples(t, focal) for t in FOLDS}
    print({t: len(v) for t, v in data.items()}, flush=True)
    rows = []
    for held in FOLDS:
        train = [s for t in FOLDS if t != held for s in data[t]]
        h0 = float(np.median([s["H"] for s in train]))
        net = train_height(train, args.epochs)
        test = data[held]
        h1 = predict_height(net, [s["crop"] for s in test], [s["aspect"] for s in test])
        for s, hh in zip(test, h1):
            rows.append({"tag": held, "stem": s["stem"], "z": s["z"], "f": s["f"], "h_px": s["h_px"],
                         "H_true": s["H"], "H_g1": float(hh), "H_g0": h0,
                         "z_g0": s["f"] * h0 / s["h_px"], "z_g1": s["f"] * float(hh) / s["h_px"],
                         "box": s["box"]})
        torch.save(net.state_dict(), OUT / f"heightnet_{held}.pt")
        print(f"{held}: H0={h0:.3f} m, n_test={len(test)}", flush=True)
    (OUT / "loso_rows.json").write_text(json.dumps(rows))
    full = train_height([s for t in FOLDS for s in data[t]], args.epochs)
    torch.save(full.state_dict(), OUT / "heightnet_all.pt")
    (OUT / "h0_all.json").write_text(json.dumps({"H0": float(np.median([s["H"] for t in FOLDS for s in data[t]]))}))
    print("done", flush=True)


if __name__ == "__main__":
    main()
