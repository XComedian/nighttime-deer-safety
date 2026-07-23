#!/usr/bin/env python3
"""Convert one archived HDF5 scene to a flat YOLO set for LOSO CV.

Uniform method across all scenes: per-deer boxes from instance_segmaps (each deer
instance -> one box), median in-mask depth -> per-box distance. This matches
connected-components on single-deer scenes (crossing7/8) and correctly separates
overlapping deer on the new multi-deer scenes (9a/9b/9c/10). Label-line order and
distances.json order are kept identical (aug-off trainer pairs them by order).

Usage: loso_convert.py <scene_dir> <tag>   e.g.  loso_convert.py /data/rural_geom_straight c9a
Output: data/loso/scenes/<tag>/{images,labels}/*.{jpg,txt} + distances.json + manifest.csv
"""
from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INVALID_DEPTH = 1e6
MIN_PX = 12


def process(h5path: Path, tag: str, out: Path, writer, distances: dict) -> None:
    with h5py.File(h5path, "r") as f:
        rgb = f["colors"][:]
        depth = f["depth"][:].astype(np.float32)
        cls = f["class_segmaps"][:]
        inst = f["instance_segmaps"][:]
        attr = json.loads(f["instance_attribute_maps"][()])
    h, w = cls.shape
    stem = f"{tag}_{int(h5path.stem):05d}"
    cv2.imwrite(str(out / "images" / f"{stem}.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

    deer_ids = [a["idx"] for a in attr if a.get("category_id") == 1]
    lines = []
    for iid in deer_ids:
        region = (inst == iid) & (cls == 1)
        n_px = int(region.sum())
        if n_px < MIN_PX:
            continue
        ys, xs = np.nonzero(region)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bw, bh = x1 - x0, y1 - y0
        vd = depth[region]
        vd = vd[vd < INVALID_DEPTH]
        dist = float(np.median(vd)) if vd.size else -1.0
        cx, cy = (x0 + bw / 2) / w, (y0 + bh / 2) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        distances.setdefault(stem, []).append([round(cx, 5), round(cy, 5), round(dist, 3)])
        writer.writerow([stem, tag, f"{dist:.2f}", f"{n_px / (bw * bh):.3f}", bw, bh, n_px])
    (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    if not lines:
        writer.writerow([stem, tag, "", "", "", "", 0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", type=Path,
                        help="Scene directory containing frames/*.hdf5 or the HDF5 files directly.")
    parser.add_argument("tag", help="Short scene tag used in the paper, such as c8 or c9a.")
    parser.add_argument("--out-root", type=Path, default=ROOT / "data/loso/scenes")
    args = parser.parse_args()
    src = args.scene_dir / "frames" if (args.scene_dir / "frames").is_dir() else args.scene_dir
    tag = args.tag
    out = args.out_root / tag
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    frames = sorted(src.glob("*.hdf5"), key=lambda p: int(p.stem))
    if not frames:
        raise FileNotFoundError(f"no HDF5 under {src}")
    distances: dict[str, list] = {}
    with (out / "manifest.csv").open("w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["frame", "scene", "distance_m", "vis_ratio", "bbox_w", "bbox_h", "mask_px"])
        for fr in frames:
            process(fr, tag, out, writer, distances)
    (out / "distances.json").write_text(json.dumps(distances))
    n_box = sum(len(v) for v in distances.values())
    print(f"{tag}: {len(frames)} frames, {n_box} boxes -> {out}")


if __name__ == "__main__":
    main()
