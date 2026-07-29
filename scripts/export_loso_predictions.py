#!/usr/bin/env python3
"""Export matched per-instance predictions from a frozen LOSO checkpoint."""
from __future__ import annotations

import argparse
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


def load_gt(scene: Path) -> dict[str, list[dict]]:
    distances = json.loads((scene / "distances.json").read_text())
    ground_truth: dict[str, list[dict]] = {}
    for image_path in sorted((scene / "images").glob("*.jpg")):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        labels = (scene / "labels" / f"{image_path.stem}.txt").read_text().splitlines()
        scene_distances = distances.get(image_path.stem, [])
        instances = []
        for index, line in enumerate(labels):
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, box_width, box_height = (float(value) for value in parts[1:])
            instances.append({
                "box": [
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    (cx + box_width / 2) * width,
                    (cy + box_height / 2) * height,
                ],
                "distance_m": float(scene_distances[index][2]),
            })
        ground_truth[image_path.stem] = instances
    return ground_truth


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ground_truth = load_gt(args.scene)
    device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    yolo_device = args.device if device.type == "cuda" else "cpu"
    distance_model = load_dist_model(device, args.weights)
    detector = YOLO(str(args.weights))
    rows = []

    for image_path in sorted((args.scene / "images").glob("*.jpg")):
        frame_index = int(image_path.stem.split("_")[-1])
        instances = ground_truth[image_path.stem]
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        letterboxed = letterbox_map(image, 640, cv2.INTER_LINEAR)
        tensor = (torch.from_numpy(cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB))
                  .permute(2, 0, 1).float().div(255)[None].to(device))
        distance_model(tensor)
        distance_map = distance_model.dist_head(distance_model._feat["p3"])

        result = detector(
            str(image_path), device=yolo_device, imgsz=640, conf=0.25,
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
        gt_boxes = np.asarray([item["box"] for item in instances], dtype=np.float32)
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
            detected = ground_truth_index in matched
            prediction = ""
            confidence = ""
            if detected:
                prediction_index = matched[ground_truth_index]
                box = predicted_boxes[prediction_index]
                grid_x, grid_y = original_center_to_grid(
                    float((box[0] + box[2]) / 2),
                    float((box[1] + box[3]) / 2),
                    width,
                    height,
                )
                grid = torch.tensor(
                    [[[[grid_x, grid_y]]]], device=device, dtype=distance_map.dtype,
                )
                prediction = float(np.exp(
                    F.grid_sample(distance_map, grid, align_corners=False).item()
                ))
                confidence = float(confidences[prediction_index])
            rows.append({
                "frame_index": frame_index,
                "instance_index": ground_truth_index,
                "distance_m": item["distance_m"],
                "detected": int(detected),
                "detection_confidence": confidence,
                "pred_m": prediction,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=list(rows[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} instances to {args.output}")


if __name__ == "__main__":
    main()
