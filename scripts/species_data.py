#!/usr/bin/env python3
"""Build balanced training crops for the size-group species classifier.

Groups: 0 deer-family, 1 horse, 2 cattle, 3 sheep, 4 dog/cat.
Sources: COCO train2017 YOLO labels (cat 15, dog 16, horse 17, sheep 18, cow 19) and the unified real deer set
(class 0 deer, day and night road imagery). Crops use the same square, 10%-context crop as HeightNet, stored at 112 px.
Train and validation crops come from different source images (COCO val2017 / unified val split for validation).
"""
from __future__ import annotations

import random
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COCO = Path(os.environ.get("COCO_ROOT", ROOT / "data/coco_2017"))
DEER = Path(os.environ.get("REAL_DEER_ROOT", ROOT / "data/real_deer_unified"))
OUT = ROOT / "output/species"
GROUPS = ["deer", "horse", "cattle", "sheep", "dog_cat"]
COCO_MAP = {15: 4, 16: 4, 17: 1, 18: 3, 19: 2}
SIZE = 112
PER_GROUP = {"train": 3000, "val": 400}
MIN_BOX_PX = 12


def crop(img, box, ctx=0.10):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - ctx * w, x1 + ctx * w, y0 - ctx * h, y1 + ctx * h
    side = max(x1 - x0, y1 - y0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    M = np.array([[SIZE / side, 0, SIZE / 2 - cx * SIZE / side], [0, SIZE / side, SIZE / 2 - cy * SIZE / side]])
    return cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)


def yolo_boxes(label_path: Path, w: int, h: int):
    for line in label_path.read_text().split("\n"):
        p = line.split()
        if len(p) != 5:
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
        yield c, [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]


def collect(split: str) -> tuple[np.ndarray, np.ndarray]:
    rnd = random.Random(0 if split == "train" else 1)
    cand = {g: [] for g in range(len(GROUPS))}
    coco_split = "train2017" if split == "train" else "val2017"
    labels = sorted((COCO / "labels" / coco_split).glob("*.txt"))
    for lp in labels:
        txt = lp.read_text()
        if not any(f"\n{c} " in "\n" + txt for c in COCO_MAP):
            continue
        cand_img = COCO / "images" / coco_split / (lp.stem + ".jpg")
        for c, box in yolo_boxes(lp, 1, 1):
            if c in COCO_MAP:
                cand[COCO_MAP[c]].append((cand_img, lp, box))
    for lp in sorted((DEER / "labels" / split).glob("*.txt")):
        for c, box in yolo_boxes(lp, 1, 1):
            if c == 0:
                cand[0].append((DEER / "images" / split / (lp.stem + ".jpg"), lp, box))
    xs, ys = [], []
    for g, items in cand.items():
        rnd.shuffle(items)
        kept = 0
        for img_path, lp, nbox in items:
            if kept >= PER_GROUP[split]:
                break
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            box = [nbox[0] * w, nbox[1] * h, nbox[2] * w, nbox[3] * h]
            if box[3] - box[1] < MIN_BOX_PX or box[2] - box[0] < MIN_BOX_PX:
                continue
            xs.append(crop(img, box)); ys.append(g); kept += 1
        print(split, GROUPS[g], "candidates", len(items), "kept", kept, flush=True)
    return np.stack(xs), np.array(ys)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        x, y = collect(split)
        np.savez_compressed(OUT / f"crops_{split}.npz", x=x, y=y)


if __name__ == "__main__":
    main()
