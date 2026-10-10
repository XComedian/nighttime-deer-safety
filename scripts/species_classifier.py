#!/usr/bin/env python3
"""Train the size-group species classifier (ImageNet ResNet-18 on 112-px crops) and evaluate it on ZOD.

Augmentation makes training crops resemble distant, dark dashcam animals for every group (random downscale to
10-112 px, blur, gamma darkening, sensor noise, grayscale, flips) so that image style does not identify the group.
ZOD evaluation: tagged animals (hoofed + dogs) in the 70- and 54.4-degree views; reindeer and deer-or-reindeer
count as deer-family. Range uses the predicted group's height: leave-one-frame-out mean LiDAR box height of that
group on ZOD (hard = top-1 group; soft = probability-weighted log-height).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/species"
GROUPS = ["deer", "horse", "cattle", "sheep", "dog_cat"]
ZOD_TO_GROUP = {"reindeer": 0, "deer_or_reindeer": 0, "horse": 1, "cattle": 2, "sheep": 3}
DEV = "cuda:0"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def degrade(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,112,112) RGB in [0,1] on device."""
    b = x.shape[0]
    out = []
    for i in range(b):
        xi = x[i:i + 1]
        s = int(np.random.uniform(10, 112))
        xi = F.interpolate(F.interpolate(xi, size=(s, s), mode="area"), size=(112, 112), mode="bilinear", align_corners=False)
        out.append(xi)
    x = torch.cat(out)
    if np.random.rand() < 0.5:
        x = x.flip(3)
    gamma = torch.empty(b, 1, 1, 1, device=x.device).uniform_(1.0, 3.0)
    dark = torch.rand(b, 1, 1, 1, device=x.device) < 0.5
    x = torch.where(dark, x.clamp(1e-4, 1) ** gamma, x)
    x = x * torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.6, 1.3)
    gray = torch.rand(b, 1, 1, 1, device=x.device) < 0.2
    x = torch.where(gray, x.mean(1, keepdim=True).expand_as(x), x)
    x = x + torch.randn_like(x) * torch.empty(b, 1, 1, 1, device=x.device).uniform_(0, 0.06)
    return x.clamp(0, 1)


def to_t(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr[..., ::-1].copy()).permute(0, 3, 1, 2).float().div(255)


def model() -> nn.Module:
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(512, len(GROUPS))
    return m.to(DEV)


@torch.no_grad()
def predict(m, crops: np.ndarray) -> np.ndarray:
    m.eval(); out = []
    for i in range(0, len(crops), 256):
        x = to_t(crops[i:i + 256]).to(DEV)
        out.append(F.softmax(m((x - MEAN.to(DEV)) / STD.to(DEV)), 1).cpu().numpy())
    return np.concatenate(out)


def train(epochs: int = 10) -> nn.Module:
    tr = np.load(OUT / "crops_train.npz"); va = np.load(OUT / "crops_val.npz")
    xtr, ytr = tr["x"], torch.tensor(tr["y"])
    m = model(); torch.manual_seed(0); np.random.seed(0)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = epochs * ((len(xtr) + 127) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.15)
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(len(xtr))
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            x = degrade(to_t(xtr[idx]).to(DEV))
            loss = F.cross_entropy(m((x - MEAN.to(DEV)) / STD.to(DEV)), ytr[idx].to(DEV), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc = (predict(m, va["x"]).argmax(1) == va["y"]).mean()
        print(f"epoch {ep}: val acc {acc:.3f}", flush=True)
    torch.save(m.state_dict(), OUT / "species_resnet18.pt")
    return m


def zod_eval(m) -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from species_data import crop as crop112
    ev = json.loads((ROOT / "output/depth_baselines/eval_boxes.json").read_text())
    res = {}
    for fov in ("70", "54.4"):
        d = pd.DataFrame(ev[f"zod_{fov}"])
        d = d[(d.tag == "ungulate") | (d.tag == "dog")].reset_index(drop=True)
        d["group"] = [4 if t == "dog" else ZOD_TO_GROUP[s] for t, s in zip(d.tag, d.species)]
        crops = np.stack([crop112(cv2.imread(r.image), r.box) for r in d.itertuples()])
        p = predict(m, crops)
        d["pred"] = p.argmax(1)
        d["h_px"] = d.box.map(lambda b: b[3] - b[1])
        d["H_true"] = d.gt_z * d.h_px / d.fx
        # leave-one-frame-out group mean of H_true (box-effective height)
        Hg = np.zeros((len(d), len(GROUPS)))
        for gi in range(len(GROUPS)):
            for i, r in d.iterrows():
                pool = d[(d.group == gi) & (d.frame != r.frame)].H_true
                Hg[i, gi] = pool.mean() if len(pool) else d[d.group == gi].H_true.mean()
        d["H_hard"] = Hg[np.arange(len(d)), d.pred]
        d["H_soft"] = np.exp((p * np.log(Hg)).sum(1))
        d["H_oraclegroup"] = Hg[np.arange(len(d)), d.group]
        conf = np.zeros((len(GROUPS), len(GROUPS)), int)
        for g, q in zip(d.group, d.pred):
            conf[g, q] += 1
        out = {"n": int(len(d)), "accuracy": float((d.pred == d.group).mean()),
               "accuracy_hoofed": float((d.pred == d.group)[d.tag == "ungulate"].mean()),
               "confusion_rows_true_cols_pred": conf.tolist()}
        hoof = d[d.tag == "ungulate"]
        for name, col in (("G1", "z_g1"), ("group_hard", "H_hard"), ("group_soft", "H_soft"), ("group_oracle", "H_oraclegroup"),
                          ("size_oracle", "z_oracle")):
            z = hoof[col] if col.startswith("z_") else hoof.fx * hoof[col] / hoof.h_px
            rel = np.abs(z - hoof.gt_z) / hoof.gt_z
            out[name] = {"median_rel": float(rel.median()), "within_10pct": float((rel <= 0.10).mean())}
        res[fov] = out
        d.drop(columns=["box"]).to_csv(OUT / f"zod_species_rows_hfov{fov}.csv", index=False)
        print(f"== ZOD {fov} deg: n={len(d)} acc {out['accuracy']:.2f} (hoofed {out['accuracy_hoofed']:.2f})")
        print("   confusion (rows true, cols pred) ", GROUPS); [print("  ", GROUPS[i].ljust(8), row) for i, row in enumerate(conf.tolist())]
        for k in ("G1", "group_hard", "group_soft", "group_oracle", "size_oracle"):
            print(f"   hoofed range error {k:13s} {100*out[k]['median_rel']:.1f}%  within10% {out[k]['within_10pct']:.2f}")
    (OUT / "zod_species_eval.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    m = train()
    zod_eval(m)
