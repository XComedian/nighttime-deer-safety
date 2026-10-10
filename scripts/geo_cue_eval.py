#!/usr/bin/env python3
"""Road-furniture ablation for the geometry-anchored range: the four cue scenes (identical except markings and
posts) evaluated zero-shot with the same checkpoint as the original ablation (Rural Conifer fold), for the
learned range head and G1 (HeightNet from the same fold). Focal recovered as in recover_synthetic_focal.py."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from eval_synthetic_distance import iou_one_to_many, letterbox_map, load_dist_model, original_center_to_grid
from geo_range import OUT, HeightNet, crop_box, predict_height
from loso_eval import load_gt
from recover_synthetic_focal import SYN, frame_k

ROOT = Path(__file__).resolve().parents[1]
CUES = {"cueA": "markings + posts", "cueB": "markings only", "cueC": "posts only", "cueD": "neither"}
WIDTH = json.loads((ROOT / "output/synthetic_focal/report.json").read_text())["road_width_m"]
BANDS = [("30-40", 30, 40), ("40-50", 40, 50), ("50-60", 50, 60)]
DEVICE = "cuda:0"


@torch.no_grad()
def main():
    ckpt = ROOT / "output/loso_asset/c14/weights/best.pt"
    dist_model = load_dist_model(torch.device(DEVICE), ckpt); det = YOLO(str(ckpt))
    net = HeightNet().to(DEVICE); net.load_state_dict(torch.load(OUT / "heightnet_c14.pt")); net.eval()
    out = {}
    for cue, label in CUES.items():
        paths = sorted((SYN / cue).glob("*.hdf5"), key=lambda p: int(p.stem))
        with ProcessPoolExecutor(32) as ex:
            ks = dict(zip([int(p.stem) for p in paths], ex.map(frame_k, paths, chunksize=10)))
        recs = []
        for stem, items in sorted(load_gt(cue).items()):
            items = [g for g in items if g["distance_m"] > 0]
            k = ks.get(int(stem.split("_")[1]))
            if not items or k is None:
                continue
            f = k / WIDTH
            img_path = ROOT / "data/loso/scenes" / cue / "images" / f"{stem}.jpg"
            img = cv2.imread(str(img_path)); h, w = img.shape[:2]
            lb = letterbox_map(img, 640, cv2.INTER_LINEAR)
            t = torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(DEVICE)
            dist_model(t); dmap = dist_model.dist_head(dist_model._feat["p3"])
            r = det(str(img_path), device=DEVICE, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
            pb = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4)); cf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
            gtb = np.array([g["box"] for g in items], np.float32); used, match = set(), {}
            for pi in np.argsort(-cf):
                ious = iou_one_to_many(pb[pi].astype(np.float32), gtb)
                for gi in used: ious[gi] = -1
                gi = int(np.argmax(ious))
                if ious[gi] >= 0.5: used.add(gi); match[gi] = int(pi)
            for gi, g in enumerate(items):
                rec = {"z": g["distance_m"], "det": gi in match}
                if gi in match:
                    b = pb[match[gi]]
                    gx, gy = original_center_to_grid((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, w, h)
                    rec["head"] = float(np.exp(F.grid_sample(dmap, torch.tensor([[[[gx, gy]]]], dtype=dmap.dtype, device=DEVICE), align_corners=False).item()))
                    hh = predict_height(net, [crop_box(img, b)], [(b[2] - b[0]) / max(b[3] - b[1], 1e-3)])[0]
                    rec["g1"] = f * float(hh) / max(b[3] - b[1], 1.0)
                recs.append(rec)
        res = {}
        for band, lo, hi in BANDS:
            bb = [x for x in recs if lo <= x["z"] < hi]; m = [x for x in bb if x["det"]]
            res[band] = {"recall": len(m) / len(bb)}
            for meth in ("head", "g1"):
                e = np.array([x[meth] - x["z"] for x in m])
                res[band][meth] = {"mae": float(np.abs(e).mean()), "joint": float((np.abs(e) <= 5).sum() / len(bb))}
        out[cue] = res
        print(f"{cue} ({label}):", "  ".join(f"{b}: R {res[b]['recall']:.2f} head {res[b]['head']['mae']:.2f}m/{res[b]['head']['joint']:.2f}  G1 {res[b]['g1']['mae']:.2f}m/{res[b]['g1']['joint']:.2f}" for b, _, _ in BANDS), flush=True)
    (OUT / "cue_ablation_geo.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
