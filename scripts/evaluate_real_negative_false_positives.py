#!/usr/bin/env python3
"""Count deer detections on the 16 deer-absent dusk frames."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
NEGATIVE_DIR = ROOT / "data" / "real_negatives"
OUTPUT_DIR = ROOT / "output" / "real_negative_false_positives"
CONFIDENCE = 0.25
NMS_IOU = 0.70
IMAGE_SIZE = 640
BATCH_SIZE = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    global NEGATIVE_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-dir", type=Path, default=NEGATIVE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model entry as display_name=/path/to/checkpoint.pt; repeat for each model.",
    )
    args = parser.parse_args()
    NEGATIVE_DIR = args.negative_dir
    OUTPUT_DIR = args.output_dir
    models = {
        item.split("=", 1)[0]: Path(item.split("=", 1)[1])
        for item in args.model
    }
    images = sorted(NEGATIVE_DIR.glob("*.jpg"))
    if len(images) != 16:
        raise RuntimeError(f"Expected 16 negative frames, found {len(images)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summaries: list[dict] = []

    for model_name, checkpoint in models.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = YOLO(str(checkpoint))
        deer_ids = [class_id for class_id, name in model.names.items() if name.lower() == "deer"]
        if len(deer_ids) != 1:
            raise RuntimeError(f"{model_name}: expected one deer class, found {deer_ids}")
        deer_id = deer_ids[0]

        detections = model.predict(
            source=[str(path) for path in images],
            device="cpu",
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            rect=False,
            verbose=False,
        )
        for image, result in zip(images, detections, strict=True):
            if result.boxes is None or len(result.boxes) == 0:
                confidences: list[float] = []
            else:
                confidences = [
                    float(confidence)
                    for class_id, confidence in zip(
                        result.boxes.cls.cpu().tolist(),
                        result.boxes.conf.cpu().tolist(),
                        strict=True,
                    )
                    if int(class_id) == deer_id
                ]
            rows.append(
                {
                    "model": model_name,
                    "frame": image.name,
                    "deer_detections": len(confidences),
                    "max_deer_confidence": (
                        f"{max(confidences):.6f}" if confidences else ""
                    ),
                }
            )

        model_rows = [row for row in rows if row["model"] == model_name]
        total = sum(int(row["deer_detections"]) for row in model_rows)
        alert_frames = sum(int(row["deer_detections"]) > 0 for row in model_rows)
        summaries.append(
            {
                "model": model_name,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "negative_frames": len(images),
                "false_positive_detections": total,
                "frames_with_false_positive": alert_frames,
                "false_positives_per_frame": total / len(images),
                "frame_alert_rate": alert_frames / len(images),
            }
        )

    fieldnames = ["model", "frame", "deer_detections", "max_deer_confidence"]
    with (OUTPUT_DIR / "per_frame.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    artifact = {
        "protocol": {
            "negative_frame_directory": str(NEGATIVE_DIR),
            "negative_frame_manifest_sha256": hashlib.sha256(
                "\n".join(f"{path.name},{sha256(path)}" for path in images).encode()
            ).hexdigest(),
            "target_class": "deer",
            "confidence_threshold": CONFIDENCE,
            "nms_iou": NMS_IOU,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "square_letterbox": True,
            "device": "cpu",
        },
        "models": summaries,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# False positives on 16 deer-absent dusk frames",
        "",
        (
            f"Frozen inference at confidence {CONFIDENCE:.2f}, NMS IoU {NMS_IOU:.2f}, "
            f"and square {IMAGE_SIZE}-pixel input (batch {BATCH_SIZE}). Only predictions "
            "assigned to the deer class "
            "are counted."
        ),
        "",
        "| model | deer detections | frames alerted | FP/frame | frame alert rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['model']} | {summary['false_positive_detections']} | "
            f"{summary['frames_with_false_positive']}/16 | "
            f"{summary['false_positives_per_frame']:.3f} | "
            f"{summary['frame_alert_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "These 16 frames come from four temporally related self-collected clips. "
                "The counts are a nuisance-output diagnostic, not a deployment false-alarm rate."
            ),
        ]
    )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")

    for summary in summaries:
        print(
            f"{summary['model']}: {summary['false_positive_detections']} deer detections, "
            f"{summary['frames_with_false_positive']}/16 frames alerted"
        )


if __name__ == "__main__":
    main()
