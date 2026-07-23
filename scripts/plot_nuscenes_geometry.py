#!/usr/bin/env python3
"""Create the real-geometry comparison figure and paired frame summary."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from nuscenes_geometry_experiment import (
    ROOT,
    load_geo_model,
    load_head_checkpoint,
    preprocess,
    read_manifest,
    sample_map,
)


DEFAULT_OUT = ROOT / "output/nuscenes_geometry"
METRICS = {
    "depth_absrel": "Depth AbsRel",
    "depth_rmse_m": "Depth RMSE (m)",
    "depth_delta125": "Depth delta < 1.25",
    "normal_angular_error_deg": "Normal error (degrees)",
}


def paired_intervals(out: Path, seed: int = 0, draws: int = 10000) -> dict:
    def load(name: str) -> list[dict]:
        with (out / name / "frame_metrics.csv").open() as f:
            return list(csv.DictReader(f))

    baseline, adapted = load("baseline_test"), load("adapted_test")
    rng, result = np.random.default_rng(seed), {}
    for key, label in METRICS.items():
        b = np.array([float(row[key]) for row in baseline])
        a = np.array([float(row[key]) for row in adapted])
        valid = np.isfinite(b) & np.isfinite(a)
        # Positive improvement always means better.
        diff = (a - b if key == "depth_delta125" else b - a)[valid]
        scenes = np.array([row["scene"] for row, keep in zip(baseline, valid) if keep])
        unique_scenes = np.unique(scenes)
        scene_means = np.array([diff[scenes == scene].mean() for scene in unique_scenes])
        samples = scene_means[rng.integers(0, len(scene_means), size=(draws, len(scene_means)))].mean(1)
        result[key] = {
            "label": label,
            "paired_frames": int(len(diff)),
            "independent_test_scenes": int(len(unique_scenes)),
            "mean_frame_improvement": float(diff.mean()),
            "scene_cluster_bootstrap_95ci": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        }
    return result


def write_scene_metrics(out: Path) -> list[dict]:
    grouped = {}
    for model in ("baseline", "adapted"):
        with (out / f"{model}_test/frame_metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for scene in sorted({row["scene"] for row in rows}):
            selected = [row for row in rows if row["scene"] == scene]
            weights = np.array([float(row["depth_points"]) for row in selected])
            values = {key: np.array([float(row[key]) for row in selected]) for key in METRICS}
            grouped.setdefault(scene, {"scene": scene, "frames": len(selected)})
            grouped[scene][f"{model}_depth_absrel"] = float(np.average(values["depth_absrel"], weights=weights))
            grouped[scene][f"{model}_depth_rmse_m"] = float(np.sqrt(np.average(values["depth_rmse_m"] ** 2, weights=weights)))
            grouped[scene][f"{model}_depth_delta125"] = float(np.average(values["depth_delta125"], weights=weights))
            normal = values["normal_angular_error_deg"]
            grouped[scene][f"{model}_normal_angular_error_deg"] = float(np.nanmean(normal))
    output = list(grouped.values())
    with (out / "test_scene_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    return output


@torch.no_grad()
def predictions(model, row: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    tensor, uv, depth_gt, road_uv, normal_gt = preprocess(row, device)
    model(tensor)
    feat = model._f["p3"]
    depth = sample_map(model.depth_head(feat), uv, device).flatten().clamp(0.1, 80).cpu().numpy()
    angle = float("nan")
    if len(road_uv) >= 50 and np.isfinite(normal_gt).all():
        normal_map = F.normalize(model.normal_head(feat), dim=1)
        normal = sample_map(normal_map, road_uv, device)
        cosine = (normal * torch.from_numpy(normal_gt).to(device)).sum(1).clamp(-1, 1)
        angle = float(torch.rad2deg(torch.arccos(cosine)).mean().cpu())
    return uv, depth, angle


def overlay(ax, image: np.ndarray, uv: np.ndarray, depth: np.ndarray, title: str) -> None:
    ax.imshow(image)
    points = ax.scatter(uv[:, 0], uv[:, 1], c=depth, s=2, cmap="turbo", vmin=0, vmax=60)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return points


def make_figure(out: Path, device: torch.device) -> Path:
    rows = read_manifest(out, "test")
    chosen = [rows[index] for index in (5, len(rows) // 2, len(rows) - 5)]
    baseline = load_geo_model(device).eval()
    adapted = load_geo_model(device).eval()
    load_head_checkpoint(adapted, out / "adapted/best_heads.pt")

    fig, axes = plt.subplots(3, 4, figsize=(12, 7.4), constrained_layout=True)
    color_points = None
    for row_index, row in enumerate(chosen):
        image = cv2.cvtColor(cv2.imread(str(ROOT / row["image"])), cv2.COLOR_BGR2RGB)
        target = np.load(ROOT / row["target"])
        uv_b, depth_b, angle_b = predictions(baseline, row, device)
        uv_a, depth_a, angle_a = predictions(adapted, row, device)
        depth_gt = target["depth_m"].astype(np.float32)
        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(f"{row['scene']}, frame {row['frame']}", fontsize=9)
        axes[row_index, 0].axis("off")
        color_points = overlay(axes[row_index, 1], image, target["uv"], depth_gt, "Projected LiDAR reference")
        absrel_b = np.mean(np.abs(depth_b - depth_gt) / depth_gt)
        absrel_a = np.mean(np.abs(depth_a - depth_gt) / depth_gt)
        overlay(axes[row_index, 2], image, uv_b, depth_b, f"Synthetic only: AbsRel {absrel_b:.2f}, normal {angle_b:.1f} deg")
        overlay(axes[row_index, 3], image, uv_a, depth_a, f"Adapted: AbsRel {absrel_a:.2f}, normal {angle_a:.1f} deg")
    colorbar = fig.colorbar(color_points, ax=axes, shrink=0.82, pad=0.01)
    colorbar.set_label("Depth (m)")
    path = out / "nuscenes_real_geometry_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intervals = paired_intervals(out)
    (out / "paired_frame_intervals.json").write_text(json.dumps(intervals, indent=2))
    baseline = json.loads((out / "baseline_test/summary.json").read_text())
    adapted = json.loads((out / "adapted_test/summary.json").read_text())
    figure = make_figure(out, device)
    scene_metrics = write_scene_metrics(out)
    lines = [
        "# nuScenes Night Geometry Experiment",
        "",
        "Acquisition-log-disjoint nighttime protocol; the final test log contains five scenes.",
        "",
        "| Metric | Synthetic only | Adapted | Relative change |",
        "|---|---:|---:|---:|",
    ]
    for key, label in METRICS.items():
        b, a = baseline[key], adapted[key]
        change = (a - b) / b if key == "depth_delta125" else (b - a) / b
        lines.append(f"| {label} | {b:.3f} | {a:.3f} | {100 * change:+.1f}% |")
    lines += [
        "",
        "Intervals use a paired scene-cluster bootstrap over the five test scenes; they remain conditional on one acquisition log.",
        "",
        f"Depth AbsRel improved in {sum(row['adapted_depth_absrel'] < row['baseline_depth_absrel'] for row in scene_metrics)}/{len(scene_metrics)} test scenes; scene-level heterogeneity is retained in `test_scene_metrics.csv`.",
        "",
        "Normals are LiDAR-derived RANSAC road-plane pseudo-references. Grade is a local road-to-ego tilt proxy; no absolute surveyed road grade is evaluated.",
        "",
        f"Figure: `{figure.relative_to(ROOT)}`",
    ]
    (out / "experiment_summary.md").write_text("\n".join(lines) + "\n")
    print(figure)


if __name__ == "__main__":
    main()
