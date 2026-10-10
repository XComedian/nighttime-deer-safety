#!/usr/bin/env python3
"""Fine-tune HeightNet on real ZOD animals with drive-grouped 5-fold cross-validation.

Target per real animal (70-degree virtual view): H* = Z_LiDAR * h_box / f, the height that makes Z = f H / h exact.
Statues are excluded. Frames are grouped into drives (same collection car, gaps <= 10 min); folds split drives,
so herd members and neighboring frames never straddle train and test.
Variants on identical folds: rendered-only (no fine-tuning), rendered->real fine-tuned, real-only from scratch.
Test predictions are made for the same animals in the 70- and 54.4-degree views.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from geo_range import OUT, HeightNet, crop_box, predict_height, to_tensor

ROOT = Path(__file__).resolve().parents[1]
ZOD = Path(os.environ.get("ZOD_ROOT", ROOT / "data/zod")) / "single_frames"
DEV = "cuda:0"
N_FOLDS = 5


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = json.loads((ROOT / "output/depth_baselines/eval_boxes.json").read_text())
    a, b = pd.DataFrame(ev["zod_70"]), pd.DataFrame(ev["zod_54.4"])
    a = a[a.tag != "statue"].reset_index(drop=True)
    meta = {f: json.loads((ZOD / f / "metadata.json").read_text()) for f in a.frame.unique()}
    fr = pd.DataFrame({"frame": list(meta), "car": [meta[f]["collection_car"] for f in meta],
                       "t": pd.to_datetime([meta[f]["time"] for f in meta])}).sort_values(["car", "t"])
    drive, g, prev = [], -1, None
    for _, r in fr.iterrows():
        if prev is None or r.car != prev.car or (r.t - prev.t).total_seconds() > 600:
            g += 1
        drive.append(g); prev = r
    fr["drive"] = drive
    a = a.merge(fr[["frame", "drive"]], on="frame")
    drives = sorted(a.drive.unique())
    rng = np.random.default_rng(0); rng.shuffle(drives)
    fold_of = {dv: i % N_FOLDS for i, dv in enumerate(drives)}
    a["fold"] = a.drive.map(fold_of)
    a["h_px"] = a.box.map(lambda bx: bx[3] - bx[1])
    a["H_star"] = a.gt_z * a.h_px / a.fx
    a["key"] = a.frame + "_" + a.gt_z.round(2).astype(str)
    b["key"] = b.frame + "_" + b.gt_z.round(2).astype(str)
    b = b[b.key.isin(a.key)].merge(a[["key", "fold"]], on="key")
    for df in (a, b):
        cache = {}
        crops = []
        for _, r in df.iterrows():
            if r.image not in cache:
                cache = {r.image: cv2.imread(r.image)}
            crops.append(crop_box(cache[r.image], r.box))
        df["crop"] = crops
        df["aspect"] = df.box.map(lambda bx: (bx[2] - bx[0]) / max(bx[3] - bx[1], 1e-3))
    return a, b


def train(df: pd.DataFrame, init: str | None, epochs: int, lr: float, seed: int = 0, pool: str = "all", balance: bool = False) -> HeightNet:
    torch.manual_seed(seed); np.random.seed(seed)
    if pool == "large":
        df = df[df.H_star >= 0.8]          # deer-scale animals only (selection on training labels)
    w = np.ones(len(df))
    if balance:                             # equal weight per log-height bin
        bins = np.digitize(np.log(df.H_star.to_numpy()), np.linspace(np.log(0.25), np.log(2.5), 9))
        counts = np.bincount(bins, minlength=bins.max() + 1)
        w = 1.0 / counts[bins]; w = w / w.mean()
    wt = torch.tensor(w, dtype=torch.float32)
    net = HeightNet().to(DEV)
    if init:
        net.load_state_dict(torch.load(init))
    x = to_tensor(list(df.crop)); asp = torch.tensor(df.aspect.to_numpy(), dtype=torch.float32)
    y = torch.log(torch.tensor(df.H_star.to_numpy(), dtype=torch.float32))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(x) + 63) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 64):
            idx = perm[i:i + 64]
            xb = x[idx].to(DEV)
            if np.random.rand() < 0.5:
                xb = xb.flip(3)
            xb = (xb * torch.empty(len(xb), 1, 1, 1, device=DEV).uniform_(0.6, 1.4)).clamp(0, 1)
            loss = (F.smooth_l1_loss(net(xb, asp[idx].to(DEV)), y[idx].to(DEV), reduction="none") * wt[idx].to(DEV)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval()


def main() -> None:
    a, b = build()
    print(f"train pool {len(a)} animals, {a.frame.nunique()} frames, {a.drive.nunique()} drives; hoofed {int((a.tag == 'ungulate').sum())}", flush=True)
    rendered = OUT / "heightnet_all.pt"
    variants = {"rendered_only": None, "finetuned": dict(init=str(rendered), epochs=15, lr=3e-4),
                "real_only": dict(init=None, epochs=40, lr=2e-3),
                "finetuned_large": dict(init=str(rendered), epochs=30, lr=3e-4, pool="large"),
                "finetuned_balanced": dict(init=str(rendered), epochs=15, lr=3e-4, balance=True)}
    for name in variants:
        a[f"H_{name}"] = np.nan; b[f"H_{name}"] = np.nan
    base = HeightNet().to(DEV); base.load_state_dict(torch.load(rendered)); base.eval()
    for k in range(N_FOLDS):
        tr, te_a, te_b = a[a.fold != k], a.fold == k, b.fold == k
        for name, cfg in variants.items():
            net = base if cfg is None else train(tr, **cfg)
            a.loc[te_a, f"H_{name}"] = predict_height(net, list(a[te_a].crop), list(a[te_a].aspect))
            b.loc[te_b, f"H_{name}"] = predict_height(net, list(b[te_b].crop), list(b[te_b].aspect))
        print(f"fold {k}: train {len(tr)}, test {int(te_a.sum())}", flush=True)
    final = train(a, **variants["finetuned"])
    torch.save(final.state_dict(), OUT / "heightnet_zod_finetuned.pt")
    cols = ["frame", "gt_z", "tag", "species", "time_of_day", "fold", "drive", "fx", "h_px", "H_star"] + [f"H_{n}" for n in variants]
    a[cols].to_csv(OUT / "zod_finetune_rows_hfov70.csv", index=False)
    b = b.assign(h_px=b.box.map(lambda bx: bx[3] - bx[1]))
    b[["frame", "gt_z", "tag", "species", "time_of_day", "fold", "fx", "h_px"] + [f"H_{n}" for n in variants]].to_csv(
        OUT / "zod_finetune_rows_hfov54.4.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
