#!/usr/bin/env python3
"""Training-scene-calibrated LOSO evaluation for deer detection and ranging.

For one outer LOSO fold, this script:
1. runs the frozen fold checkpoint on its existing inner-validation frames;
2. selects a compact log-range calibration model by leave-one-training-scene-out
   validation within those frames;
3. refits the selected calibrator on all inner-validation matches; and
4. evaluates it once on the held-out outer scene.

The outer scene is never used to fit or select the calibrator.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from loso_eval import BANDS, TOL, load_gt  # noqa: E402

CONF = 0.25
IOU_MATCH = 0.5
RIDGE = 1e-3
CANDIDATES = ("raw", "raw_box", "raw_box_pos")


def feature_row(record: dict, candidate: str) -> list[float]:
    row = [np.log(max(record["raw_range_m"], 1e-3))]
    if candidate in {"raw_box", "raw_box_pos"}:
        row.extend([
            np.log(max(record["box_h_frac"], 1e-5)),
            np.log(max(record["box_w_frac"], 1e-5)),
        ])
    if candidate == "raw_box_pos":
        row.append(record["center_y_frac"])
    return row


class RidgeCalibrator:
    def __init__(self, candidate: str):
        self.candidate = candidate
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.beta: np.ndarray | None = None

    def fit(self, records: list[dict]) -> "RidgeCalibrator":
        x = np.asarray([feature_row(r, self.candidate) for r in records], dtype=float)
        y = np.log(np.asarray([r["gt_range_m"] for r in records], dtype=float))
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        z = (x - self.mean) / self.scale
        design = np.column_stack([np.ones(len(z)), z])
        penalty = np.eye(design.shape[1]) * RIDGE
        penalty[0, 0] = 0.0
        self.beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, records: list[dict]) -> np.ndarray:
        assert self.mean is not None and self.scale is not None and self.beta is not None
        x = np.asarray([feature_row(r, self.candidate) for r in records], dtype=float)
        z = (x - self.mean) / self.scale
        design = np.column_stack([np.ones(len(z)), z])
        return np.exp(design @ self.beta)

    def as_dict(self) -> dict:
        assert self.mean is not None and self.scale is not None and self.beta is not None
        return {
            "candidate": self.candidate,
            "feature_mean": self.mean.tolist(),
            "feature_scale": self.scale.tolist(),
            "coefficients": self.beta.tolist(),
        }


@torch.no_grad()
def match_records(paths: list[Path], weights: Path, device: str) -> tuple[list[dict], dict[str, int]]:
    dist_model = load_dist_model(torch.device(device), weights)
    detector = YOLO(str(weights))
    gt_cache: dict[str, dict[str, list[dict]]] = {}
    records: list[dict] = []
    counts = defaultdict(int)

    for image_path in paths:
        tag = image_path.parent.parent.name
        if tag not in gt_cache:
            gt_cache[tag] = load_gt(tag)
        items = gt_cache[tag].get(image_path.stem, [])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        h, w = image.shape[:2]

        lb = letterbox_map(image, 640, cv2.INTER_LINEAR)
        tensor = (torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))
                  .permute(2, 0, 1).float().div(255)[None].to(device))
        dist_model(tensor)
        dmap = dist_model.dist_head(dist_model._feat["p3"])

        result = detector(
            str(image_path), device=device, imgsz=640, conf=CONF, iou=0.7, verbose=False
        )[0]
        pred_boxes = (
            result.boxes.xyxy.cpu().numpy().astype(np.float32)
            if len(result.boxes) else np.zeros((0, 4), np.float32)
        )
        confidence = (
            result.boxes.conf.cpu().numpy().astype(np.float32)
            if len(result.boxes) else np.zeros(0, np.float32)
        )
        gt_boxes = np.asarray([item["box"] for item in items], dtype=np.float32)
        matched: dict[int, int] = {}
        used: set[int] = set()
        for pred_i in np.argsort(-confidence):
            if not len(gt_boxes):
                break
            ious = iou_one_to_many(pred_boxes[pred_i], gt_boxes)
            for gt_i in used:
                ious[gt_i] = -1
            gt_i = int(np.argmax(ious))
            if ious[gt_i] >= IOU_MATCH:
                used.add(gt_i)
                matched[gt_i] = int(pred_i)

        for gt_i, item in enumerate(items):
            distance = float(item["distance_m"])
            band = next((name for name, lo, hi in BANDS if lo <= distance < hi), None)
            if band is None:
                continue
            counts[f"gt_{band}"] += 1
            if gt_i not in matched:
                continue
            counts[f"det_{band}"] += 1
            box = pred_boxes[matched[gt_i]]
            center_x = float((box[0] + box[2]) / 2)
            center_y = float((box[1] + box[3]) / 2)
            grid_x, grid_y = original_center_to_grid(center_x, center_y, w, h)
            grid = torch.tensor([[[[grid_x, grid_y]]]], device=device, dtype=dmap.dtype)
            raw_range = float(np.exp(
                F.grid_sample(dmap, grid, align_corners=False).item()
            ))
            records.append({
                "scene": tag,
                "frame": image_path.stem,
                "band": band,
                "gt_range_m": distance,
                "raw_range_m": raw_range,
                "box_h_frac": float((box[3] - box[1]) / h),
                "box_w_frac": float((box[2] - box[0]) / w),
                "center_y_frac": center_y / h,
            })
    return records, dict(counts)


def candidate_score(records: list[dict], candidate: str) -> dict:
    scenes = sorted({r["scene"] for r in records})
    errors: list[float] = []
    hits = 0
    total = 0
    for scene in scenes:
        train = [r for r in records if r["scene"] != scene]
        test = [r for r in records if r["scene"] == scene and r["band"] == "50-60"]
        if not train or not test:
            continue
        pred = RidgeCalibrator(candidate).fit(train).predict(test)
        err = np.abs(pred - np.asarray([r["gt_range_m"] for r in test]))
        errors.extend(err.tolist())
        hits += int((err <= TOL).sum())
        total += len(err)
    return {
        "candidate": candidate,
        "inner_scenes": len(scenes),
        "n_50_60": total,
        "mae_50_60_m": float(np.mean(errors)) if errors else float("nan"),
        "within_5m_50_60": hits / total if total else float("nan"),
    }


def evaluate(records: list[dict], counts: dict[str, int], predictions: np.ndarray) -> dict:
    out: dict[str, dict] = {}
    for band, _, _ in BANDS:
        band_indices = [i for i, record in enumerate(records) if record["band"] == band]
        gt = counts.get(f"gt_{band}", 0)
        det = counts.get(f"det_{band}", 0)
        if band_indices:
            target = np.asarray([records[i]["gt_range_m"] for i in band_indices])
            pred = predictions[band_indices]
            error = np.abs(pred - target)
            mae = float(error.mean())
            joint = int((error <= TOL).sum()) / gt if gt else float("nan")
        else:
            mae = joint = float("nan")
        out[band] = {
            "gt": gt,
            "detected": det,
            "recall": det / gt if gt else float("nan"),
            "matched_mae_m": mae,
            "joint_within_5m": joint,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "output/loso_asset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/loso_calibrated")
    args = parser.parse_args()

    weights = args.checkpoint_root / args.fold / "weights/best.pt"
    fold_dir = ROOT / "data/loso/folds" / args.fold
    if not weights.exists():
        raise FileNotFoundError(weights)
    inner_paths = [Path(line) for line in (fold_dir / "val.txt").read_text().splitlines()]
    outer_paths = sorted((ROOT / "data/loso/scenes" / args.fold / "images").glob("*.jpg"))

    inner_records, _ = match_records(inner_paths, weights, args.device)
    scores = [candidate_score(inner_records, candidate) for candidate in CANDIDATES]
    chosen = sorted(
        scores,
        key=lambda row: (-row["within_5m_50_60"], row["mae_50_60_m"], row["candidate"]),
    )[0]["candidate"]
    calibrator = RidgeCalibrator(chosen).fit(inner_records)

    outer_records, outer_counts = match_records(outer_paths, weights, args.device)
    raw_predictions = np.asarray([r["raw_range_m"] for r in outer_records])
    calibrated_predictions = calibrator.predict(outer_records)
    result = {
        "fold": args.fold,
        "selection_protocol": "leave-one-training-scene-out on inner-validation matches",
        "inner_match_count": len(inner_records),
        "candidate_scores": scores,
        "calibrator": calibrator.as_dict(),
        "raw": evaluate(outer_records, outer_counts, raw_predictions),
        "calibrated": evaluate(outer_records, outer_counts, calibrated_predictions),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.fold}.json"
    output_path.write_text(json.dumps(result, indent=2))
    band = result["calibrated"]["50-60"]
    print(
        f"{args.fold}: chose {chosen}; 50-60 m calibrated "
        f"MAE={band['matched_mae_m']:.2f} joint={band['joint_within_5m']:.3f}"
    )
    print(output_path)


if __name__ == "__main__":
    main()
