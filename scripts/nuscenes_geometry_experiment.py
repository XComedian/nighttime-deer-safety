#!/usr/bin/env python3
"""Evaluate and adapt synthetic geometry heads on nuScenes night scenes.

References are sensor-derived: sparse camera depth from projected LiDAR and a
camera-frame road normal from a RANSAC plane fit. The full protocol separates
acquisition logs so adjacent scenes never cross train, validation, and test.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_multitask_geometry import Decoder, FeatureHook, GeoModel, letterbox_map  # noqa: E402

import __main__  # noqa: E402

for cls in (Decoder, FeatureHook, GeoModel):
    setattr(__main__, cls.__name__, cls)

ARCH = ROOT / "configs/detection_range_model.yaml"
SYNTH_CKPT = ROOT / "output/synthetic/searched_geo/weights/last.pt"
MINI_SPLITS = {"train": ["scene-1077"], "val": ["scene-1094"], "test": ["scene-1100"]}
FULL_VAL_LOG = "c9b039c47ec54cc7b5c0fcc7a5730e38"
FULL_TEST_LOG = "f93e8d66ce4b4fbea7062d19b1fe29fb"


def quat_matrix(q: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / max(n, 1e-12)
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ], dtype=np.float64)


def load_json(path: Path):
    try:
        import orjson
        return orjson.loads(path.read_bytes())
    except ImportError:
        return json.loads(path.read_text())


def iter_json_array(path: Path, chunk_size: int = 1 << 20):
    """Stream objects from a top-level JSON array without loading the full table."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer, started, finished = "", False, False
        while not finished:
            chunk = handle.read(chunk_size)
            if not chunk:
                finished = True
            buffer += chunk
            position = 0
            while True:
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] in "[,\n"):
                    if buffer[position] == "[":
                        started = True
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                if not started or position >= len(buffer):
                    break
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                yield value
                position = end
            buffer = buffer[position:]
        if buffer.strip() not in {"", "]"}:
            raise ValueError(f"Incomplete JSON array: {path}")


def resolve_splits(root: Path, protocol: str) -> tuple[str, dict[str, list[str]]]:
    if protocol == "mini":
        return "v1.0-mini", MINI_SPLITS
    scenes = load_json(root / "v1.0-trainval/scene.json")
    part10 = scenes[-85:]
    if len(part10) != 85 or not all("night" in row["description"].lower() for row in part10):
        raise ValueError("Expected the final 85 trainval metadata scenes to be night blob part 10")
    splits = {"train": [], "val": [], "test": []}
    for row in part10:
        split = "test" if row["log_token"] == FULL_TEST_LOG else "val" if row["log_token"] == FULL_VAL_LOG else "train"
        splits[split].append(row["name"])
    return "v1.0-trainval", splits


class NuScenesData:
    def __init__(self, root: Path, version: str, scene_names: set[str]):
        self.root = root
        meta = root / version
        self.tables = {}
        for name in ["scene", "sample", "calibrated_sensor", "sensor"]:
            rows = load_json(meta / f"{name}.json")
            self.tables[name] = {row["token"]: row for row in rows}
        self.scenes = {row["name"]: row for row in self.tables["scene"].values() if row["name"] in scene_names}
        needed_samples = set()
        for scene_name in scene_names:
            token = self.scenes[scene_name]["first_sample_token"]
            while token:
                needed_samples.add(token)
                token = self.tables["sample"][token]["next"]
        self.sample_channels = {}
        sample_data = {}
        for sd in iter_json_array(meta / "sample_data.json"):
            if sd["sample_token"] not in needed_samples or not sd["is_key_frame"]:
                continue
            cal = self.tables["calibrated_sensor"][sd["calibrated_sensor_token"]]
            channel = self.tables["sensor"][cal["sensor_token"]]["channel"]
            if channel not in {"CAM_FRONT", "LIDAR_TOP"}:
                continue
            sample_data[sd["token"]] = sd
            self.sample_channels.setdefault(sd["sample_token"], {})[channel] = sd["token"]
        self.tables["sample_data"] = sample_data
        needed_poses = {row["ego_pose_token"] for row in sample_data.values()}
        self.tables["ego_pose"] = {
            row["token"]: row for row in iter_json_array(meta / "ego_pose.json") if row["token"] in needed_poses
        }

    def scene_samples(self, name: str) -> list[dict]:
        scene = self.scenes[name]
        out, token = [], scene["first_sample_token"]
        while token:
            row = self.tables["sample"][token]
            out.append(row)
            token = row["next"]
        return out

    def sensor_record(self, sample: dict, channel: str) -> tuple[dict, dict, dict]:
        sd = self.tables["sample_data"][self.sample_channels[sample["token"]][channel]]
        cal = self.tables["calibrated_sensor"][sd["calibrated_sensor_token"]]
        pose = self.tables["ego_pose"][sd["ego_pose_token"]]
        return sd, cal, pose


def transform_points(points: np.ndarray, rotation: list[float], translation: list[float]) -> np.ndarray:
    return points @ quat_matrix(rotation).T + np.asarray(translation, dtype=np.float64)


def inverse_transform_points(points: np.ndarray, rotation: list[float], translation: list[float]) -> np.ndarray:
    return (points - np.asarray(translation, dtype=np.float64)) @ quat_matrix(rotation)


def fit_road_plane(points_ego: np.ndarray, seed: int, iterations: int = 300,
                   threshold: float = 0.16) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points_ego) < 80:
        return None
    rng = np.random.default_rng(seed)
    best = np.zeros(len(points_ego), dtype=bool)
    for _ in range(iterations):
        tri = points_ego[rng.choice(len(points_ego), 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        if abs(normal[2]) < 0.75:
            continue
        dist = np.abs((points_ego - tri[0]) @ normal)
        mask = dist < threshold
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < 50:
        return None
    center = points_ego[best].mean(axis=0)
    _, _, vt = np.linalg.svd(points_ego[best] - center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    dist = np.abs((points_ego - center) @ normal)
    mask = dist < threshold
    return normal / np.linalg.norm(normal), mask


def unique_nearest_pixels(uv: np.ndarray, depth: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    pix = np.rint(uv).astype(np.int32)
    order = np.argsort(depth)
    pix, depth = pix[order], depth[order]
    linear = pix[:, 1].astype(np.int64) * width + pix[:, 0]
    _, keep = np.unique(linear, return_index=True)
    return pix[keep].astype(np.float32), depth[keep].astype(np.float32)


def prepare(root: Path, out: Path, version: str, split_scenes: dict[str, list[str]]) -> None:
    all_scenes = {scene for scenes in split_scenes.values() for scene in scenes}
    nusc = NuScenesData(root, version, all_scenes)
    target_dir = out / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, scene_names in split_scenes.items():
      for scene_name in scene_names:
        for frame_idx, sample in enumerate(nusc.scene_samples(scene_name)):
            cam_sd, cam_cal, cam_pose = nusc.sensor_record(sample, "CAM_FRONT")
            lid_sd, lid_cal, lid_pose = nusc.sensor_record(sample, "LIDAR_TOP")
            image_path = root / cam_sd["filename"]
            lidar_path = root / lid_sd["filename"]
            image = cv2.imread(str(image_path))
            if image is None or not lidar_path.exists():
                continue
            height, width = image.shape[:2]
            raw = np.fromfile(lidar_path, dtype=np.float32)
            if raw.size % 5:
                raise ValueError(f"Unexpected LiDAR layout: {lidar_path}")
            xyz_lidar = raw.reshape(-1, 5)[:, :3].astype(np.float64)
            xyz_ego_lidar = transform_points(xyz_lidar, lid_cal["rotation"], lid_cal["translation"])
            xyz_global = transform_points(xyz_ego_lidar, lid_pose["rotation"], lid_pose["translation"])
            xyz_ego_cam = inverse_transform_points(xyz_global, cam_pose["rotation"], cam_pose["translation"])
            xyz_cam = inverse_transform_points(xyz_ego_cam, cam_cal["rotation"], cam_cal["translation"])

            depth = xyz_cam[:, 2]
            intrinsic = np.asarray(cam_cal["camera_intrinsic"], dtype=np.float64)
            projected = xyz_cam @ intrinsic.T
            uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-9)
            valid = ((depth > 2.0) & (depth < 60.0) & (uv[:, 0] >= 1) & (uv[:, 0] < width - 1)
                     & (uv[:, 1] >= 1) & (uv[:, 1] < height - 1))
            all_idx = np.flatnonzero(valid)
            uv_depth, sparse_depth = unique_nearest_pixels(uv[all_idx], depth[all_idx], width)

            road_candidates = valid & (uv[:, 0] > 0.12 * width) & (uv[:, 0] < 0.88 * width)
            road_candidates &= (uv[:, 1] > 0.55 * height) & (xyz_ego_cam[:, 0] > 3.0)
            candidate_idx = np.flatnonzero(road_candidates)
            scene_seed = int(scene_name.split("-")[-1])
            fit = fit_road_plane(xyz_ego_cam[candidate_idx], seed=frame_idx + 1000 * scene_seed)
            if fit is None:
                road_uv = np.zeros((0, 2), dtype=np.float32)
                normal_cam = np.full(3, np.nan, dtype=np.float32)
                normal_ego = np.full(3, np.nan, dtype=np.float32)
                grade_pct = float("nan")
                plane_inliers = 0
            else:
                normal_ego, inlier_local = fit
                inlier_idx = candidate_idx[inlier_local]
                road_uv, _ = unique_nearest_pixels(uv[inlier_idx], depth[inlier_idx], width)
                r_cam = quat_matrix(cam_cal["rotation"])
                normal_cam = r_cam.T @ normal_ego
                normal_cam /= np.linalg.norm(normal_cam)
                # BlenderProc synthetic targets orient the visible road normal toward +camera-y.
                if normal_cam[1] < 0:
                    normal_cam = -normal_cam
                grade_pct = float(-100.0 * normal_ego[0] / max(normal_ego[2], 1e-6))
                plane_inliers = len(road_uv)
                if abs(grade_pct) > 10.0:
                    road_uv = np.zeros((0, 2), dtype=np.float32)
                    normal_cam = np.full(3, np.nan, dtype=np.float32)
                    normal_ego = np.full(3, np.nan, dtype=np.float32)
                    grade_pct = float("nan")
                    plane_inliers = 0

            target_path = target_dir / f"{scene_name}_{frame_idx:03d}.npz"
            np.savez_compressed(
                target_path,
                uv=uv_depth,
                depth_m=sparse_depth,
                road_uv=road_uv,
                normal_cam=np.asarray(normal_cam, dtype=np.float32),
                normal_ego=np.asarray(normal_ego, dtype=np.float32),
                grade_pct=np.float32(grade_pct),
            )
            rows.append({
                "split": split,
                "scene": scene_name,
                "frame": frame_idx,
                "sample_token": sample["token"],
                "image": str(image_path.relative_to(ROOT)),
                "target": str(target_path.relative_to(ROOT)),
                "depth_points": len(uv_depth),
                "plane_inliers": plane_inliers,
                "grade_pct": grade_pct,
            })

    with (out / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {split: sum(r["split"] == split for r in rows) for split in split_scenes}
    valid_planes = sum(r["plane_inliers"] >= 50 for r in rows)
    preview_dir = out / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for split in split_scenes:
        row = next(r for r in rows if r["split"] == split and r["plane_inliers"] >= 50)
        image = cv2.imread(str(ROOT / row["image"]))
        target = np.load(ROOT / row["target"])
        uv, depth = target["uv"], target["depth_m"]
        keep = np.arange(0, len(uv), max(1, len(uv) // 1800))
        colors = cv2.applyColorMap(
            np.clip(255 * (depth[keep] - 2.0) / 58.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
        ).reshape(-1, 3)
        for point, color in zip(uv[keep].astype(int), colors):
            cv2.circle(image, tuple(point), 2, tuple(int(x) for x in color), -1, cv2.LINE_AA)
        road_uv = target["road_uv"]
        road_keep = np.arange(0, len(road_uv), max(1, len(road_uv) // 700))
        for point in road_uv[road_keep].astype(int):
            cv2.circle(image, tuple(point), 2, (40, 220, 40), -1, cv2.LINE_AA)
        cv2.imwrite(str(preview_dir / f"{split}_{row['scene']}.jpg"), image)
    (out / "inventory.md").write_text(
        f"# nuScenes {version} Night Geometry Inventory\n\n"
        + "\n".join(
            f"- {split}: {len(split_scenes[split])} scenes, {summary[split]} frames" for split in split_scenes
        )
        + f"\n- Frames with valid road-plane fits: {valid_planes}/{len(rows)}\n"
        + "- Depth reference: projected single-sweep LiDAR (sparse).\n"
        + "- Normal reference: deterministic RANSAC road-plane fit with a +/-10% local-tilt quality gate.\n"
        + "- Reported grade is a local road-to-ego tilt proxy, not absolute surveyed road grade.\n"
    )
    print(f"prepared {len(rows)} frames; valid planes {valid_planes}; wrote {out}")


def load_geo_model(device: torch.device) -> GeoModel:
    ckpt = torch.load(SYNTH_CKPT, map_location="cpu", weights_only=False)
    trained = ckpt["model"].float()
    model = GeoModel(str(ARCH))
    model.load_state_dict(trained.state_dict(), strict=False)
    return model.eval().to(device)


def load_head_checkpoint(model: GeoModel, path: Path | None) -> None:
    if path is None:
        return
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.depth_head.load_state_dict(state["depth_head"])
    model.normal_head.load_state_dict(state["normal_head"])


def read_manifest(out: Path, split: str) -> list[dict]:
    with (out / "manifest.csv").open(newline="") as f:
        return [row for row in csv.DictReader(f) if row["split"] == split]


def preprocess(row: dict, device: torch.device) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(str(ROOT / row["image"]))
    h, w = image.shape[:2]
    lb = letterbox_map(image, 640, cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    target = np.load(ROOT / row["target"])
    scale = min(640 / h, 640 / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    top, left = (640 - nh) // 2, (640 - nw) // 2
    uv = target["uv"].astype(np.float32) * scale + np.array([left, top], dtype=np.float32)
    road_uv = target["road_uv"].astype(np.float32) * scale + np.array([left, top], dtype=np.float32)
    return tensor, uv, target["depth_m"].astype(np.float32), road_uv, target["normal_cam"].astype(np.float32)


def pixel_grid(uv: np.ndarray, device: torch.device) -> torch.Tensor:
    x = 2.0 * (torch.from_numpy(uv[:, 0]).to(device) + 0.5) / 640.0 - 1.0
    y = 2.0 * (torch.from_numpy(uv[:, 1]).to(device) + 0.5) / 640.0 - 1.0
    return torch.stack([x, y], dim=1).view(1, 1, -1, 2)


def sample_map(pred: torch.Tensor, uv: np.ndarray, device: torch.device) -> torch.Tensor:
    if not len(uv):
        return torch.empty(0, device=device)
    return F.grid_sample(pred, pixel_grid(uv, device), align_corners=False).squeeze(0).squeeze(1).T


@torch.no_grad()
def evaluate(model: GeoModel, rows: list[dict], device: torch.device) -> tuple[list[dict], dict]:
    metrics, all_pred, all_gt, all_cos = [], [], [], []
    for row in rows:
        tensor, uv, depth_gt, road_uv, normal_gt = preprocess(row, device)
        model(tensor)
        feat = model._f["p3"]
        depth_map = model.depth_head(feat)
        normal_map = F.normalize(model.normal_head(feat), dim=1)
        depth_pred = sample_map(depth_map, uv, device).flatten().clamp(0.1, 80).cpu().numpy()
        valid = np.isfinite(depth_gt) & (depth_gt > 0.5)
        p, g = depth_pred[valid], depth_gt[valid]
        absrel = float(np.mean(np.abs(p - g) / g)) if len(g) else float("nan")
        rmse = float(np.sqrt(np.mean((p - g) ** 2))) if len(g) else float("nan")
        delta = float(np.mean(np.maximum(p / g, g / np.maximum(p, 1e-6)) < 1.25)) if len(g) else float("nan")
        all_pred.append(p)
        all_gt.append(g)
        angle = float("nan")
        if len(road_uv) >= 50 and np.isfinite(normal_gt).all():
            normal_pred = sample_map(normal_map, road_uv, device)
            ng = torch.from_numpy(normal_gt).to(device)
            cos = (normal_pred * ng).sum(1).clamp(-1, 1)
            angle = float(torch.rad2deg(torch.arccos(cos)).mean().cpu())
            all_cos.append(cos.cpu().numpy())
        metrics.append({
            "scene": row["scene"], "frame": row["frame"], "depth_points": len(g),
            "depth_absrel": absrel, "depth_rmse_m": rmse, "depth_delta125": delta,
            "normal_points": len(cos) if len(road_uv) >= 50 and np.isfinite(normal_gt).all() else 0,
            "normal_angular_error_deg": angle, "grade_pct_reference": row["grade_pct"],
        })
    p = np.concatenate(all_pred) if all_pred else np.array([])
    g = np.concatenate(all_gt) if all_gt else np.array([])
    cos = np.concatenate(all_cos) if all_cos else np.array([])
    summary = {
        "frames": len(rows), "depth_points": len(g),
        "depth_absrel": float(np.mean(np.abs(p - g) / g)),
        "depth_rmse_m": float(np.sqrt(np.mean((p - g) ** 2))),
        "depth_delta125": float(np.mean(np.maximum(p / g, g / np.maximum(p, 1e-6)) < 1.25)),
        "normal_points": len(cos),
        "normal_angular_error_deg": float(np.mean(np.degrees(np.arccos(np.clip(cos, -1, 1))))) if len(cos) else float("nan"),
    }
    return metrics, summary


def write_eval(out: Path, name: str, rows: list[dict], summary: dict) -> None:
    eval_dir = out / name
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "frame_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (eval_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    (eval_dir / "summary.md").write_text(
        f"# {name.replace('_', ' ').title()}\n\n"
        f"- Frames: {summary['frames']}\n- Sparse LiDAR depth points: {summary['depth_points']}\n"
        f"- Depth AbsRel: {summary['depth_absrel']:.3f}\n- Depth RMSE: {summary['depth_rmse_m']:.2f} m\n"
        f"- Depth delta < 1.25: {summary['depth_delta125']:.3f}\n"
        f"- Road-normal angular error: {summary['normal_angular_error_deg']:.1f} degrees\n\n"
        "References are projected single-sweep LiDAR depth and RANSAC road-plane normals; they are sensor-derived, not manual dense ground truth.\n"
    )


def train(out: Path, device: torch.device, epochs: int, lr: float,
          train_stride: int = 1, selection_stride: int = 1) -> Path:
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model = load_geo_model(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for head in (model.depth_head, model.normal_head):
        head.train()
        for parameter in head.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        list(model.depth_head.parameters()) + list(model.normal_head.parameters()), lr=lr, weight_decay=1e-4
    )
    all_train_rows, all_val_rows = read_manifest(out, "train"), read_manifest(out, "val")
    train_rows = [row for row in all_train_rows if int(row["frame"]) % train_stride == 0]
    val_rows = [row for row in all_val_rows if int(row["frame"]) % selection_stride == 0]
    ckpt_dir = out / "adapted"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path, best_score, history = ckpt_dir / "best_heads.pt", float("inf"), []
    for epoch in range(1, epochs + 1):
        random.Random(epoch).shuffle(train_rows)
        losses = []
        for row in train_rows:
            tensor, uv, depth_gt, road_uv, normal_gt = preprocess(row, device)
            with torch.no_grad():
                model(tensor); feat = model._f["p3"].detach()
            depth_map = model.depth_head(feat)
            normal_map = F.normalize(model.normal_head(feat), dim=1)
            if len(uv) > 2500:
                keep = np.random.default_rng(epoch * 1000 + int(row["frame"])).choice(len(uv), 2500, replace=False)
                uv, depth_gt = uv[keep], depth_gt[keep]
            pred_depth = sample_map(depth_map, uv, device).flatten().clamp(1e-3)
            gt_depth = torch.from_numpy(depth_gt).to(device)
            depth_loss = F.smooth_l1_loss(torch.log(pred_depth), torch.log(gt_depth))
            normal_loss = torch.tensor(0.0, device=device)
            if len(road_uv) >= 50 and np.isfinite(normal_gt).all():
                if len(road_uv) > 1000:
                    keep = np.random.default_rng(epoch * 2000 + int(row["frame"])).choice(len(road_uv), 1000, replace=False)
                    road_uv = road_uv[keep]
                pred_normal = sample_map(normal_map, road_uv, device)
                gt_normal = torch.from_numpy(normal_gt).to(device)
                normal_loss = (1.0 - (pred_normal * gt_normal).sum(1)).mean()
            loss = depth_loss + normal_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        _, val = evaluate(model, val_rows, device)
        score = val["depth_absrel"] + val["normal_angular_error_deg"] / 90.0
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_score": score, **val})
        if score < best_score:
            best_score = score
            torch.save({
                "depth_head": model.depth_head.state_dict(), "normal_head": model.normal_head.state_dict(),
                "source": str(SYNTH_CKPT),
                "train_scenes": sorted({row["scene"] for row in all_train_rows}),
                "val_scenes": sorted({row["scene"] for row in all_val_rows}),
                "epoch": epoch, "val_score": score,
            }, best_path)
        print(f"epoch {epoch:03d} loss={np.mean(losses):.4f} val_absrel={val['depth_absrel']:.3f} val_normal={val['normal_angular_error_deg']:.1f}", flush=True)
        for head in (model.depth_head, model.normal_head):
            head.train()
    with (ckpt_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    return best_path


def run(args: argparse.Namespace) -> None:
    version, split_scenes = resolve_splits(args.root, args.protocol)
    if args.out is None:
        args.out = ROOT / ("output/nuscenes_geometry" if args.protocol == "mini" else "output/nuscenes_geometry_full")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode in {"prepare", "all"}:
        prepare(args.root, args.out, version, split_scenes)
    device = torch.device(args.device)
    if args.mode in {"baseline", "all"}:
        model = load_geo_model(device)
        for split in split_scenes:
            rows, summary = evaluate(model, read_manifest(args.out, split), device)
            write_eval(args.out, f"baseline_{split}", rows, summary)
            print(f"baseline {split}: {summary}")
    if args.mode in {"adapt", "all"}:
        best_path = train(args.out, device, args.epochs, args.lr, args.train_stride, args.selection_stride)
        model = load_geo_model(device); load_head_checkpoint(model, best_path)
        for split in ("val", "test"):
            rows, summary = evaluate(model, read_manifest(args.out, split), device)
            write_eval(args.out, f"adapted_{split}", rows, summary)
            print(f"adapted {split}: {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/nuscenes")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--protocol", choices=["mini", "full-night"], default="mini")
    parser.add_argument("--mode", choices=["prepare", "baseline", "adapt", "all"], default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--selection-stride", type=int, default=1)
    run(parser.parse_args())
