#!/usr/bin/env python3
"""Privileged DISTANCE multi-task on the searched detector (on-theme, deployable).

A distance-regression head on the P3 neck feature predicts a per-pixel distance map;
supervised ONLY at GT deer-box centers with the metric distance (from synthetic depth).
At inference: for each detected box, read the map at its center -> per-deer distance (m).

Alignment trick: box centers are read from the AUGMENTED batch (ultralytics tracks them
in letterboxed coords); distances paired by order from distances.json (geometric aug OFF
-> box order preserved). No dense-map letterbox sync needed. Head is a light add-on;
inference detector stays ~2.01M.
"""
import argparse, json
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "configs/detection_range_model.yaml"
DIST_JSON = {}

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

P3_LAYER = 16


class FeatureHook:
    """Picklable forward hook storing a layer's output into a shared dict."""
    def __init__(self, d, key):
        self.d, self.key = d, key

    def __call__(self, m, i, o):
        self.d[self.key] = o


class DistHead(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, 64, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 1, 1))  # log-distance

    def forward(self, x):
        return self.net(x)


class DistModel(DetectionModel):
    def __init__(self, cfg, nc=1, verbose=False):
        super().__init__(cfg, nc=nc, verbose=verbose)
        self._feat = {}
        self._hook = self.model[P3_LAYER].register_forward_hook(FeatureHook(self._feat, "p3"))
        with torch.no_grad():
            self.forward(torch.zeros(1, 3, 640, 640))
        cin = self._feat["p3"].shape[1]
        self.dist_head = DistHead(cin)
        self.dist_w = 2.0

    def __getstate__(self):
        self._feat.clear()
        st = self.__dict__.copy()
        st["_hook"] = None  # RemovableHandle isn't picklable
        return st

    def loss(self, batch, preds=None):
        det_loss, items = super().loss(batch, preds)
        feat = self._feat.get("p3")
        if feat is None or not self.training:   # dist aux only during training
            return det_loss, items
        self.dist_head.to(feat.device)
        dmap = self.dist_head(feat)  # (N,1,h,w) log-distance
        N, _, h, w = dmap.shape
        bi = batch["batch_idx"].view(-1)
        boxes = batch["bboxes"]  # (M,4) normalized cx,cy,w,h in letterboxed space
        losses = []
        for i in range(N):
            stem = Path(batch["im_file"][i]).stem
            gt = DIST_JSON.get(stem, [])
            sel = boxes[bi == i]
            if not gt or len(sel) == 0:
                continue
            n = min(len(gt), len(sel))            # pair by order (aug off)
            cx = sel[:n, 0] * 2 - 1               # -> grid_sample coords [-1,1]
            cy = sel[:n, 1] * 2 - 1
            grid = torch.stack([cx, cy], -1).view(1, 1, n, 2).to(dmap.device)
            pred = F.grid_sample(dmap[i:i + 1], grid, align_corners=False).view(n)
            tgt = torch.tensor([g[2] for g in gt[:n]], device=dmap.device, dtype=pred.dtype)
            losses.append(F.smooth_l1_loss(pred, torch.log(tgt + 1e-3)))
        if losses:
            dl = torch.stack(losses).mean() * self.dist_w
            det_loss = det_loss + dl * N
        return det_loss, items


class DistTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        m = DistModel(cfg or str(ARCH), nc=1, verbose=verbose)
        if weights:
            m.load(weights)
        return m


def main():
    global DIST_JSON
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--data", default=str(ROOT / "data/synthetic_yolo/data.yaml"))
    ap.add_argument("--distances-json", default=str(ROOT / "data/synthetic_yolo/distances.json"))
    ap.add_argument("--project", default=str(ROOT / "output/synthetic"))
    ap.add_argument("--name", default="searched_dist")
    ap.add_argument(
        "--weights",
        required=True,
        help="Initialization checkpoint. Use the archived paper checkpoint or a compatible COCO checkpoint.",
    )
    ap.add_argument("--patience", type=int, default=40)
    args = ap.parse_args()
    DIST_JSON = json.loads(Path(args.distances_json).read_text())
    ov = dict(model=str(ARCH), data=args.data,
              epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.gpu,
              project=args.project, name=args.name,
              seed=0, patience=args.patience, verbose=False, exist_ok=True, pretrained=args.weights,
              mosaic=0.0, close_mosaic=0, scale=0.0, degrees=0.0, shear=0.0,
              perspective=0.0, translate=0.0, fliplr=0.0, amp=False)  # aug off -> box order stable
    DistTrainer(overrides=ov).train()
    print("DONE dist")


if __name__ == "__main__":
    main()
