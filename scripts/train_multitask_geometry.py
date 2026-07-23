#!/usr/bin/env python3
"""Dense geometry heads (DEPTH + NORMALS) on the searched detector, trained on synthetic.

Adds two decoder heads on the P3 neck feature; supervised by exported depth (meters) and
normals (camera-frame, [-1,1]). Aug is OFF; the depth/normal TARGETS are letterboxed to imgsz
with the SAME ratio+pad as the image (replicated here) so they align with the network output.
Heads are extra outputs of the agent (kept at inference for geometry). Depth: SiLog on valid
pixels. Normals: 1 - cosine similarity on valid pixels.
"""
import argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "configs/detection_range_model.yaml"
AUX = ROOT / "data/synthetic_yolo/aux"

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.build import build_yolo_dataset

P3 = 16


def letterbox_map(m, imgsz=640, interp=cv2.INTER_NEAREST):
    """Replicate ultralytics default LetterBox (center pad, scaleup) on an HxW[xC] target."""
    h0, w0 = m.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    m = cv2.resize(m, (nw, nh), interpolation=interp)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    bot, right = imgsz - nh - top, imgsz - nw - left
    return cv2.copyMakeBorder(m, top, bot, left, right, cv2.BORDER_CONSTANT, value=0)


class FeatureHook:
    def __init__(self, d, k): self.d, self.k = d, k
    def __call__(self, m, i, o): self.d[self.k] = o


class Decoder(nn.Module):
    def __init__(self, cin, cout, act=None):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, 64, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, cout, 1))
        self.act = act
    def forward(self, x):
        y = self.net(x)
        return self.act(y) if self.act else y


class GeoModel(DetectionModel):
    def __init__(self, cfg, nc=1, verbose=False):
        super().__init__(cfg, nc=nc, verbose=verbose)
        self._f = {}
        self._hook = self.model[P3].register_forward_hook(FeatureHook(self._f, "p3"))
        with torch.no_grad():
            self.forward(torch.zeros(1, 3, 640, 640))
        cin = self._f["p3"].shape[1]
        self.depth_head = Decoder(cin, 1, nn.Softplus())   # meters
        self.normal_head = Decoder(cin, 3)                 # unit-normalized in loss
        self.w_depth, self.w_normal = 1.0, 1.0

    def __getstate__(self):
        self._f.clear(); st = self.__dict__.copy(); st["_hook"] = None; return st

    def loss(self, batch, preds=None):
        det, items = super().loss(batch, preds)
        f = self._f.get("p3")
        if f is None or not self.training:
            return det, items
        self.depth_head.to(f.device); self.normal_head.to(f.device)
        N = batch["img"].shape[0]
        # DEPTH
        dp = self.depth_head(f)
        dt = batch["depth"].to(f.device).float()
        dp = F.interpolate(dp, size=dt.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        vd = dt > 0
        if vd.sum() > 64:
            g = torch.log(dp.clamp(1e-3) + 1e-3)[vd] - torch.log(dt + 1e-3)[vd]
            det = det + (torch.sqrt((g ** 2).mean() - 0.5 * g.mean() ** 2 + 1e-8)) * self.w_depth * N
        # NORMALS
        npd = self.normal_head(f)
        nt = batch["normal"].to(f.device).float()                # (N,3,h,w)
        npd = F.interpolate(npd, size=nt.shape[-2:], mode="bilinear", align_corners=False)
        npd = F.normalize(npd, dim=1)
        vn = (nt.abs().sum(1) > 0.1)
        if vn.sum() > 64:
            cos = (npd * F.normalize(nt, dim=1)).sum(1)           # (N,h,w)
            det = det + (1 - cos[vn]).mean() * self.w_normal * N
        return det, items


class GeoDataset(YOLODataset):
    def __getitem__(self, i):
        s = super().__getitem__(i)
        stem = Path(self.im_files[i]).stem
        split = "val" if f"{'images/val'}" in self.im_files[i].replace("\\", "/") else "train"
        H, W = s["img"].shape[-2:]
        d = AUX / "depth" / split / (stem + ".npy")
        n = AUX / "normal" / split / (stem + ".npy")
        if d.exists():
            dm = letterbox_map(np.load(d).astype(np.float32), H)
            s["depth"] = torch.from_numpy(dm)
        else:
            s["depth"] = torch.zeros(H, W)
        if n.exists():
            nm = letterbox_map(np.load(n).astype(np.float32), H)      # HxWx3
            s["normal"] = torch.from_numpy(nm).permute(2, 0, 1)
        else:
            s["normal"] = torch.zeros(3, H, W)
        return s

    @staticmethod
    def collate_fn(batch):
        dep = torch.stack([b.pop("depth") for b in batch])
        nor = torch.stack([b.pop("normal") for b in batch])
        out = YOLODataset.collate_fn(batch)
        out["depth"], out["normal"] = dep, nor
        return out


class GeoTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        m = GeoModel(cfg or str(ARCH), nc=1, verbose=verbose)
        if weights: m.load(weights)
        return m

    def build_dataset(self, img_path, mode="train", batch=None):
        # replicate build_yolo_dataset construction but with GeoDataset
        cfg = self.args
        return GeoDataset(img_path=img_path, imgsz=cfg.imgsz, batch_size=batch,
                          augment=(mode == "train"), hyp=cfg, rect=(mode == "val"),
                          cache=cfg.cache or None, single_cls=cfg.single_cls or False,
                          stride=32, pad=0.0 if mode == "train" else 0.5,
                          prefix=f"{mode}: ", task="detect", classes=cfg.classes,
                          data=self.data, fraction=cfg.fraction if mode == "train" else 1.0)


def main():
    global AUX
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--data", type=Path, default=ROOT / "data/synthetic_yolo/data.yaml")
    ap.add_argument("--aux", type=Path, default=ROOT / "data/synthetic_yolo/aux")
    ap.add_argument("--weights", required=True,
                    help="Initialization checkpoint compatible with the supplied architecture.")
    ap.add_argument("--project", type=Path, default=ROOT / "output/synthetic")
    ap.add_argument("--name", default="searched_geo")
    args = ap.parse_args()
    AUX = args.aux
    ov = dict(model=str(ARCH), data=str(args.data),
              epochs=args.epochs, imgsz=640, batch=32, device=args.gpu,
              project=str(args.project), name=args.name,
              seed=0, patience=40, verbose=False, exist_ok=True, pretrained=args.weights,
              amp=False, mosaic=0.0, close_mosaic=0, scale=0.0, degrees=0.0, shear=0.0,
              perspective=0.0, translate=0.0, fliplr=0.0)
    GeoTrainer(overrides=ov).train()
    print("DONE geo")


if __name__ == "__main__":
    main()
