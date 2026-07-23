#!/usr/bin/env python3
"""Export current instance-level crossing8 predictions for recalibration.

The frozen crossing7 checkpoint is evaluated on the current COCO-derived
crossing8 YOLO labels. One row is written for every ground-truth deer; missed
instances retain an empty prediction so downstream joint metrics count them as
failures.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_synthetic_distance import (iou_one_to_many, letterbox_map,  # noqa: E402
                                     load_dist_model, original_center_to_grid)

SCENE = ROOT / "data" / "loso" / "scenes" / "c8"
WEIGHTS = ROOT / "runs" / "detect" / "output" / "synthetic" / \
    "searched_dist_crossing7_ft" / "weights" / "best.pt"
OUT = ROOT / "output" / "single_scene_instance_c8" / \
    "matched_distance_predictions.csv"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_gt() -> dict[str, list[dict]]:
    distances = json.loads((SCENE / "distances.json").read_text())
    ground_truth: dict[str, list[dict]] = {}
    for image_path in sorted((SCENE / "images").glob("*.jpg")):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        labels = (SCENE / "labels" / f"{image_path.stem}.txt").read_text().splitlines()
        scene_distances = distances.get(image_path.stem, [])
        instances = []
        for index, line in enumerate(labels):
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, bw, bh = (float(value) for value in parts[1:])
            instances.append({
                "box": [
                    (cx - bw / 2) * width,
                    (cy - bh / 2) * height,
                    (cx + bw / 2) * width,
                    (cy + bh / 2) * height,
                ],
                "distance_m": float(scene_distances[index][2]),
            })
        ground_truth[image_path.stem] = instances
    return ground_truth


@torch.no_grad()
def main() -> None:
    ground_truth = load_gt()
    distance_model = load_dist_model(torch.device(DEVICE), WEIGHTS)
    detector = YOLO(str(WEIGHTS))
    rows = []

    for image_path in sorted((SCENE / "images").glob("*.jpg")):
        frame_index = int(image_path.stem.split("_")[-1])
        instances = ground_truth[image_path.stem]
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        letterboxed = letterbox_map(image, 640, cv2.INTER_LINEAR)
        tensor = (torch.from_numpy(cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB))
                  .permute(2, 0, 1).float().div(255)[None].to(DEVICE))
        distance_model(tensor)
        distance_map = distance_model.dist_head(distance_model._feat["p3"])

        result = detector(
            str(image_path), device=DEVICE, imgsz=640, conf=0.25,
            iou=0.7, verbose=False,
        )[0]
        predicted_boxes = (
            result.boxes.xyxy.cpu().numpy().astype(np.float32)
            if len(result.boxes) else np.zeros((0, 4), np.float32)
        )
        confidences = (
            result.boxes.conf.cpu().numpy().astype(np.float32)
            if len(result.boxes) else np.zeros(0, np.float32)
        )
        gt_boxes = np.array([item["box"] for item in instances], np.float32)
        matched: dict[int, int] = {}
        used: set[int] = set()
        for prediction_index in np.argsort(-confidences):
            if not len(gt_boxes):
                break
            overlaps = iou_one_to_many(predicted_boxes[prediction_index], gt_boxes)
            for ground_truth_index in used:
                overlaps[ground_truth_index] = -1
            ground_truth_index = int(np.argmax(overlaps))
            if overlaps[ground_truth_index] >= 0.5:
                used.add(ground_truth_index)
                matched[ground_truth_index] = int(prediction_index)

        for ground_truth_index, item in enumerate(instances):
            prediction = ""
            if ground_truth_index in matched:
                box = predicted_boxes[matched[ground_truth_index]]
                grid_x, grid_y = original_center_to_grid(
                    float((box[0] + box[2]) / 2),
                    float((box[1] + box[3]) / 2),
                    width,
                    height,
                )
                grid = torch.tensor(
                    [[[[grid_x, grid_y]]]], device=DEVICE, dtype=distance_map.dtype,
                )
                prediction = float(np.exp(
                    F.grid_sample(distance_map, grid, align_corners=False).item()
                ))
            rows.append({
                "frame_index": frame_index,
                "instance_index": ground_truth_index,
                "distance_m": item["distance_m"],
                "detected": int(ground_truth_index in matched),
                "pred_m": prediction,
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} instances to {OUT}")


if __name__ == "__main__":
    main()
