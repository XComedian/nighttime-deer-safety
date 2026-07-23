#!/usr/bin/env python3
"""Synthetic-val distance-aware evaluation.

Outputs:
- distance-head MAE/RMSE by GT distance bucket.
- detection recall/AP50 slices by GT distance and visibility for saved YOLO checkpoints.

The slice AP treats non-slice GT as ignored: detections that match ignored GT at IoU>=0.5
are not counted as FP for the active slice.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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

from train_distance_multitask import DistHead, DistModel, FeatureHook  # noqa: E402
from train_multitask_geometry import letterbox_map  # noqa: E402

import __main__  # noqa: E402

for cls in (DistHead, DistModel, FeatureHook):
    setattr(__main__, cls.__name__, cls)

ARCH = str(ROOT / "configs/detection_range_model.yaml")
DATA_ROOT = ROOT / "data/synthetic_yolo"
DIST_JSON = {}
IMG_DIR = DATA_ROOT / "images/val"
LBL_DIR = DATA_ROOT / "labels/val"
MANIFEST = DATA_ROOT / "manifest.csv"
OUT = ROOT / "output/synthetic_eval"

DIST_BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, math.inf)]
VIS_BUCKETS = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]


def bucket_name(lo: float, hi: float, unit: str = "") -> str:
    if math.isinf(hi):
        return f">={lo:g}{unit}"
    return f"{lo:g}-{hi:g}{unit}"


def find_bucket(x: float, buckets: list[tuple[float, float]]) -> str:
    for lo, hi in buckets:
        if lo <= x < hi:
            return bucket_name(lo, hi)
    return "out"


def xywhn_to_xyxy(box: list[float], w: int, h: int) -> np.ndarray:
    cx, cy, bw, bh = box
    return np.array([(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h], dtype=np.float32)


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def original_center_to_grid(x: float, y: float, width: int, height: int, image_size: int = 640) -> tuple[float, float]:
    """Map an original-image point into the centered letterbox grid."""
    scale = min(image_size / height, image_size / width)
    resized_height = int(round(height * scale))
    resized_width = int(round(width * scale))
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    grid_x = 2.0 * (x * scale + left) / image_size - 1.0
    grid_y = 2.0 * (y * scale + top) / image_size - 1.0
    return grid_x, grid_y


def ap_from_pr(tp: np.ndarray, fp: np.ndarray, npos: int) -> float:
    if npos == 0:
        return float("nan")
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    rec = tp / max(npos, 1)
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def load_gt() -> dict[str, list[dict]]:
    manifest = defaultdict(list)
    with MANIFEST.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "val":
                manifest[row["frame"]].append(row)

    gts = {}
    for img_path in sorted(IMG_DIR.glob("*.jpg")):
        im = cv2.imread(str(img_path))
        h, w = im.shape[:2]
        rows = manifest.get(img_path.stem, [])
        labels = []
        lbl = LBL_DIR / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    labels.append([float(x) for x in parts[1:5]])
        dists = DIST_JSON.get(img_path.stem, [])
        items = []
        for i, box in enumerate(labels):
            dist = dists[i][2] if i < len(dists) else float(rows[i]["distance_m"])
            vis = float(rows[i]["vis_ratio"]) if i < len(rows) else float("nan")
            items.append({
                "box": xywhn_to_xyxy(box, w, h),
                "cx": box[0],
                "cy": box[1],
                "distance_m": float(dist),
                "vis_ratio": vis,
                "dist_bucket": find_bucket(float(dist), DIST_BUCKETS),
                "vis_bucket": find_bucket(vis, VIS_BUCKETS) if not math.isnan(vis) else "unknown",
            })
        gts[img_path.stem] = items
    return gts


def load_dist_model(device: torch.device, weights: Path) -> DistModel:
    model = DistModel(ARCH)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"].float().state_dict(), strict=False)
    return model.eval().to(device)


@torch.no_grad()
def eval_distance_head(gts: dict[str, list[dict]], device: torch.device, max_images: int | None,
                       weights: Path) -> list[dict]:
    model = load_dist_model(device, weights)
    rows = []
    for n, img_path in enumerate(sorted(IMG_DIR.glob("*.jpg"))):
        if max_images is not None and n >= max_images:
            break
        items = gts[img_path.stem]
        if not items:
            continue
        im = cv2.imread(str(img_path))
        h, w = im.shape[:2]
        lb = letterbox_map(im, 640, cv2.INTER_LINEAR)
        t = torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(device)
        model(t)
        dmap = model.dist_head(model._feat["p3"])
        for gt in items:
            center_x = float((gt["box"][0] + gt["box"][2]) / 2)
            center_y = float((gt["box"][1] + gt["box"][3]) / 2)
            grid_x, grid_y = original_center_to_grid(center_x, center_y, w, h)
            grid = torch.tensor([[[[grid_x, grid_y]]]], device=device, dtype=dmap.dtype)
            pred = float(np.exp(F.grid_sample(dmap, grid, align_corners=False).item()))
            err = pred - gt["distance_m"]
            rows.append({
                "frame": img_path.stem,
                "distance_m": gt["distance_m"],
                "pred_m": pred,
                "abs_err_m": abs(err),
                "sq_err_m2": err * err,
                "dist_bucket": gt["dist_bucket"],
                "vis_bucket": gt["vis_bucket"],
            })
    return rows


def summarize_distance(rows: list[dict]) -> list[dict]:
    out = []
    for name in [bucket_name(lo, hi) for lo, hi in DIST_BUCKETS]:
        vals = [r for r in rows if r["dist_bucket"] == name]
        if vals:
            out.append({
                "slice": "distance",
                "bucket": name,
                "n": len(vals),
                "mae_m": np.mean([r["abs_err_m"] for r in vals]),
                "rmse_m": math.sqrt(np.mean([r["sq_err_m2"] for r in vals])),
                "bias_m": np.mean([r["pred_m"] - r["distance_m"] for r in vals]),
            })
    out.append({
        "slice": "distance",
        "bucket": "all",
        "n": len(rows),
        "mae_m": np.mean([r["abs_err_m"] for r in rows]),
        "rmse_m": math.sqrt(np.mean([r["sq_err_m2"] for r in rows])),
        "bias_m": np.mean([r["pred_m"] - r["distance_m"] for r in rows]),
    })
    return out


@torch.no_grad()
def eval_matched_distance(
    gts: dict[str, list[dict]],
    device: torch.device,
    yolo_device: str,
    max_images: int | None,
    weights: Path,
) -> list[dict]:
    """Evaluate range at centers of confidence-filtered, GT-matched detections."""
    distance_model = load_dist_model(device, weights)
    detector = YOLO(str(weights))
    image_paths = sorted(IMG_DIR.glob("*.jpg"))
    if max_images is not None:
        image_paths = image_paths[:max_images]

    rows = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        letterboxed = letterbox_map(image, 640, cv2.INTER_LINEAR)
        tensor = (
            torch.from_numpy(cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1).float().div(255)[None].to(device)
        )
        distance_model(tensor)
        distance_map = distance_model.dist_head(distance_model._feat["p3"])

        result = detector(
            str(image_path), device=yolo_device, imgsz=640, conf=0.25, iou=0.7, verbose=False,
        )[0]
        boxes = result.boxes
        predicted_boxes = (
            boxes.xyxy.cpu().numpy().astype(np.float32) if len(boxes)
            else np.zeros((0, 4), dtype=np.float32)
        )
        confidences = (
            boxes.conf.cpu().numpy().astype(np.float32) if len(boxes)
            else np.zeros(0, dtype=np.float32)
        )

        frame_gts = gts[image_path.stem]
        gt_boxes = np.array([gt["box"] for gt in frame_gts], dtype=np.float32)
        matched_gt: set[int] = set()
        matches: dict[int, tuple[int, float]] = {}
        for prediction_index in np.argsort(-confidences):
            if not len(gt_boxes):
                break
            ious = iou_one_to_many(predicted_boxes[prediction_index], gt_boxes)
            for gt_index in matched_gt:
                ious[gt_index] = -1.0
            gt_index = int(np.argmax(ious))
            if ious[gt_index] >= 0.5:
                matched_gt.add(gt_index)
                matches[gt_index] = (int(prediction_index), float(ious[gt_index]))

        for gt_index, gt in enumerate(frame_gts):
            prediction = float("nan")
            confidence = float("nan")
            matched_iou = float("nan")
            detected = gt_index in matches
            if detected:
                prediction_index, matched_iou = matches[gt_index]
                box = predicted_boxes[prediction_index]
                center_x = float((box[0] + box[2]) / 2)
                center_y = float((box[1] + box[3]) / 2)
                grid_x, grid_y = original_center_to_grid(center_x, center_y, width, height)
                grid = torch.tensor(
                    [[[[grid_x, grid_y]]]], device=device, dtype=distance_map.dtype,
                )
                prediction = float(np.exp(F.grid_sample(
                    distance_map, grid, align_corners=False,
                ).item()))
                confidence = float(confidences[prediction_index])

            error = prediction - gt["distance_m"] if detected else float("nan")
            rows.append({
                "frame": image_path.stem,
                "frame_index": int(image_path.stem.split("_")[-1]),
                "distance_m": gt["distance_m"],
                "dist_bucket": gt["dist_bucket"],
                "detected": int(detected),
                "detection_confidence": confidence,
                "matched_iou": matched_iou,
                "pred_m": prediction,
                "error_m": error,
                "abs_err_m": abs(error) if detected else float("nan"),
                "within_5m": int(detected and abs(error) <= 5.0),
            })
    return rows


def summarize_matched_distance(rows: list[dict]) -> list[dict]:
    summaries = []
    bucket_names = [bucket_name(lo, hi) for lo, hi in DIST_BUCKETS]
    for name in bucket_names + ["all"]:
        values = rows if name == "all" else [row for row in rows if row["dist_bucket"] == name]
        if not values:
            continue
        matched = [row for row in values if row["detected"]]
        errors = np.array([row["error_m"] for row in matched], dtype=float)
        summaries.append({
            "bucket": name,
            "gt": len(values),
            "detected": len(matched),
            "recall": len(matched) / len(values),
            "matched_mae_m": float(np.mean(np.abs(errors))) if len(errors) else float("nan"),
            "matched_rmse_m": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float("nan"),
            "matched_bias_m": float(np.mean(errors)) if len(errors) else float("nan"),
            "joint_within_5m": sum(row["within_5m"] for row in values) / len(values),
            "conditional_within_5m": (
                sum(row["within_5m"] for row in matched) / len(matched) if matched else float("nan")
            ),
        })
    return summaries


def eval_model_slices(model_name: str, weights: Path, gts: dict[str, list[dict]], device: str,
                      max_images: int | None) -> list[dict]:
    model = YOLO(str(weights))
    pred_by_frame = {}
    img_paths = sorted(IMG_DIR.glob("*.jpg"))
    if max_images is not None:
        img_paths = img_paths[:max_images]
    for img_path in img_paths:
        result = model(str(img_path), device=device, imgsz=640, conf=0.001, iou=0.7, verbose=False)[0]
        boxes = result.boxes
        pred_by_frame[img_path.stem] = {
            "boxes": boxes.xyxy.cpu().numpy().astype(np.float32) if len(boxes) else np.zeros((0, 4), dtype=np.float32),
            "conf": boxes.conf.cpu().numpy().astype(np.float32) if len(boxes) else np.zeros(0, dtype=np.float32),
        }

    rows = []
    slice_defs = []
    for lo, hi in DIST_BUCKETS:
        slice_defs.append(("distance", bucket_name(lo, hi), lambda gt, lo=lo, hi=hi: lo <= gt["distance_m"] < hi))
    for lo, hi in VIS_BUCKETS:
        slice_defs.append(("visibility", bucket_name(lo, hi), lambda gt, lo=lo, hi=hi: lo <= gt["vis_ratio"] < hi))

    for slice_type, bucket, keep in slice_defs:
        npos = sum(1 for stem in pred_by_frame for gt in gts[stem] if keep(gt))
        if npos == 0:
            continue
        detections = []
        recall_hits = 0
        recall_total = 0
        for stem, pred in pred_by_frame.items():
            slice_gts = [gt for gt in gts[stem] if keep(gt)]
            ignore_gts = [gt for gt in gts[stem] if not keep(gt)]
            slice_boxes = np.array([gt["box"] for gt in slice_gts], dtype=np.float32)
            ignore_boxes = np.array([gt["box"] for gt in ignore_gts], dtype=np.float32)
            matched_conf025 = set()
            order025 = np.argsort(-pred["conf"])
            for pi in order025:
                if pred["conf"][pi] < 0.25 or len(slice_boxes) == 0:
                    continue
                ious = iou_one_to_many(pred["boxes"][pi], slice_boxes)
                gi = int(np.argmax(ious))
                if ious[gi] >= 0.5 and gi not in matched_conf025:
                    matched_conf025.add(gi)
            recall_hits += len(matched_conf025)
            recall_total += len(slice_boxes)

            for pi, conf in enumerate(pred["conf"]):
                detections.append({
                    "stem": stem,
                    "box": pred["boxes"][pi],
                    "conf": float(conf),
                    "ignored": bool(len(ignore_boxes) and iou_one_to_many(pred["boxes"][pi], ignore_boxes).max() >= 0.5),
                })

        detections.sort(key=lambda d: -d["conf"])
        matched = defaultdict(set)
        tp, fp = [], []
        for det in detections:
            if det["ignored"]:
                continue
            stem = det["stem"]
            slice_gts = [gt for gt in gts[stem] if keep(gt)]
            slice_boxes = np.array([gt["box"] for gt in slice_gts], dtype=np.float32)
            if len(slice_boxes) == 0:
                tp.append(0.0)
                fp.append(1.0)
                continue
            ious = iou_one_to_many(det["box"], slice_boxes)
            gi = int(np.argmax(ious))
            if ious[gi] >= 0.5 and gi not in matched[stem]:
                matched[stem].add(gi)
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)
        ap50 = ap_from_pr(np.array(tp, dtype=float), np.array(fp, dtype=float), npos)
        rows.append({
            "model": model_name,
            "slice": slice_type,
            "bucket": bucket,
            "gt": npos,
            "recall_iou50_conf025": recall_hits / max(recall_total, 1),
            "ap50": ap50,
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    dist_summary: list[dict],
    slice_rows: list[dict],
    matched_summary: list[dict],
    max_images: int | None,
) -> None:
    lines = ["# Synthetic Distance-Aware Evaluation", ""]
    if max_images is not None:
        lines.append(f"Smoke run on first {max_images} synthetic-val images.")
        lines.append("")
    lines.append("## Distance Head")
    lines.append("")
    lines.append("| bucket | n | MAE m | RMSE m | bias m |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in dist_summary:
        lines.append(f"| {r['bucket']} | {r['n']} | {r['mae_m']:.2f} | {r['rmse_m']:.2f} | {r['bias_m']:.2f} |")
    lines.append("")
    if matched_summary:
        lines.append("## End-to-End Matched Detection and Range")
        lines.append("")
        lines.append("| bucket | GT | detected | recall | matched MAE m | matched bias m | joint within 5 m |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in matched_summary:
            lines.append(
                f"| {row['bucket']} | {row['gt']} | {row['detected']} | {row['recall']:.3f} | "
                f"{row['matched_mae_m']:.2f} | {row['matched_bias_m']:.2f} | "
                f"{row['joint_within_5m']:.3f} |"
            )
        lines.append("")
    lines.append("## Detection Slices")
    lines.append("")
    lines.append("| model | slice | bucket | GT | recall@IoU50/conf0.25 | AP50 |")
    lines.append("|---|---|---|---:|---:|---:|")
    for r in slice_rows:
        lines.append(
            f"| {r['model']} | {r['slice']} | {r['bucket']} | {r['gt']} | "
            f"{r['recall_iou50_conf025']:.3f} | {r['ap50']:.3f} |"
        )
    lines.append("")
    lines.append("Notes: slice AP ignores detections matching non-slice GT at IoU>=0.5. "
                 "Distance buckets with no GT instances are omitted. The first distance table "
                 "samples ground-truth centers; the matched table samples centers of detections "
                 "matched at IoU>=0.5 and confidence>=0.25.")
    (OUT / "synthetic_distance_eval.md").write_text("\n".join(lines))


def main() -> None:
    global DATA_ROOT, DIST_JSON, IMG_DIR, LBL_DIR, MANIFEST, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dist-weights", type=Path, default=ROOT / "output/synthetic/searched_dist/weights/last.pt")
    parser.add_argument("--det-model", action="append", default=[],
                        help="Optional model entry as name=weights.pt. Defaults to searched_det and searched_seg_box.")
    parser.add_argument("--matched-model", default=None,
                        help="Optional name=weights.pt entry for end-to-end matched distance evaluation.")
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    DIST_JSON = json.loads((DATA_ROOT / "distances.json").read_text())
    IMG_DIR = DATA_ROOT / "images/val"
    LBL_DIR = DATA_ROOT / "labels/val"
    MANIFEST = DATA_ROOT / "manifest.csv"
    OUT = args.out

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    yolo_device = "0" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    gts = load_gt()

    dist_rows = eval_distance_head(gts, device, args.max_images, args.dist_weights)
    dist_summary = summarize_distance(dist_rows)
    write_csv(OUT / "distance_head_predictions.csv", dist_rows)
    write_csv(OUT / "distance_head_summary.csv", dist_summary)

    matched_summary = []
    if args.matched_model:
        _, matched_weights = args.matched_model.split("=", 1)
        matched_rows = eval_matched_distance(
            gts, device, yolo_device, args.max_images, Path(matched_weights),
        )
        matched_summary = summarize_matched_distance(matched_rows)
        write_csv(OUT / "matched_distance_predictions.csv", matched_rows)
        write_csv(OUT / "matched_distance_summary.csv", matched_summary)

    if args.det_model:
        models = {item.split("=", 1)[0]: Path(item.split("=", 1)[1]) for item in args.det_model}
    else:
        models = {
            "searched_det": ROOT / "output/synthetic/searched_det/weights/best.pt",
            "searched_seg_box": ROOT / "output/synthetic/searched_seg/weights/best.pt",
        }
    slice_rows = []
    for name, weights in models.items():
        if weights.exists():
            slice_rows.extend(eval_model_slices(name, weights, gts, yolo_device, args.max_images))
    write_csv(OUT / "detection_slices.csv", slice_rows)
    write_report(dist_summary, slice_rows, matched_summary, args.max_images)
    print(f"wrote {OUT / 'synthetic_distance_eval.md'}")


if __name__ == "__main__":
    main()
