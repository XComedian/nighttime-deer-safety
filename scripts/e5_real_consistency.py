#!/usr/bin/env python3
"""E5a: scale-free range consistency on real dusk deer clips (no metric ground truth).

The clips have no distance reference and unknown intrinsics, so only relative range can be tested.
For a rigid deer in a fixed pose, pinhole projection gives Z * h_box = f * H = constant, i.e.
log Z = c - 1 * log h_box. A correct range head (up to an unknown camera scale) should show slope -1
within each deer track; a head that ignores the animal (e.g. constant output) shows slope 0.

Range is sampled at labelled box centres (user boxes), isolating range from detection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_synthetic_distance import letterbox_map, load_dist_model, original_center_to_grid  # noqa: E402

LABELS = ROOT / "data/real_labels.json"
FRAMES = ROOT / "data/real_frames"
OUT = ROOT / "output/e5_real_consistency"
FOLDS = ["c8", "c9a", "c9b", "c9c", "c10", "c11", "c12", "c13", "c14", "c15"]  # ten rural LOSO checkpoints
PRIMARY = "c8"                     # checkpoint used in the controller replay
GATE_PX, MIN_TRACK = 40.0, 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_tracks(labels: dict) -> list[list[dict]]:
    """Greedy nearest-centre linking within each clip; frames are 0.2 s apart."""
    by_clip: dict[str, list[tuple[float, str, list]]] = {}
    for name, boxes in labels.items():
        clip, t = name.split("__t")
        by_clip.setdefault(clip, []).append((float(t[:-4]), name, boxes))
    tracks = []
    for clip, frames in by_clip.items():
        frames.sort()
        open_tracks: list[list[dict]] = []
        for t, name, boxes in frames:
            dets = [{"clip": clip, "t": t, "frame": name, "x": b[0], "y": b[1], "w": b[2], "h": b[3],
                     "cx": b[0] + b[2] / 2, "cy": b[1] + b[3] / 2} for b in boxes]
            free = set(range(len(dets)))
            for tr in open_tracks:
                last = tr[-1]
                if t - last["t"] > 0.65 or not free:
                    continue
                j = min(free, key=lambda k: np.hypot(dets[k]["cx"] - last["cx"], dets[k]["cy"] - last["cy"]))
                if np.hypot(dets[j]["cx"] - last["cx"], dets[j]["cy"] - last["cy"]) <= GATE_PX:
                    tr.append(dets[j]); free.discard(j)
            open_tracks += [[dets[k]] for k in free]
        tracks += open_tracks
    return [tr for tr in tracks if len(tr) >= MIN_TRACK]


@torch.no_grad()
def predict_ranges(weights: Path, tracks: list[list[dict]]) -> dict[tuple[str, float, float], float]:
    model = load_dist_model(DEVICE, weights)
    cache, out = {}, {}
    for tr in tracks:
        for d in tr:
            if d["frame"] not in cache:
                image = cv2.imread(str(FRAMES / d["frame"]))
                lb = letterbox_map(image, 640, cv2.INTER_LINEAR)
                t = torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(DEVICE)
                model(t)
                cache[d["frame"]] = (model.dist_head(model._feat["p3"]), image.shape[:2])
            dmap, (h, w) = cache[d["frame"]]
            gx, gy = original_center_to_grid(d["cx"], d["cy"], w, h)
            grid = torch.tensor([[[[gx, gy]]]], dtype=dmap.dtype, device=DEVICE)
            log_d = F.grid_sample(dmap, grid, align_corners=False).item()   # head predicts log distance
            out[(d["frame"], d["cx"], d["cy"])] = float(np.exp(log_d))
    return out


def track_stats(tracks, ranges) -> list[dict]:
    rows = []
    for i, tr in enumerate(tracks):
        z = np.array([ranges[(d["frame"], d["cx"], d["cy"])] for d in tr])
        h = np.array([d["h"] for d in tr], dtype=float)
        lz, lh = np.log(z), np.log(h)
        slope = float(np.polyfit(lh, lz, 1)[0]) if np.ptp(lh) > 0 else float("nan")
        r = float(np.corrcoef(lh, lz)[0, 1]) if np.ptp(lh) > 0 and np.ptp(lz) > 0 else float("nan")
        rows.append({"track": i, "clip": tr[0]["clip"], "n": len(tr), "t0": tr[0]["t"], "t1": tr[-1]["t"],
                     "h_min": float(h.min()), "h_max": float(h.max()), "h_ratio": float(h.max() / h.min()),
                     "z_min": float(z.min()), "z_max": float(z.max()), "z_median": float(np.median(z)),
                     "loglog_slope": slope, "pearson_logz_logh": r,
                     "cv_z_times_h": float(np.std(z * h) / np.mean(z * h)), "cv_h": float(np.std(h) / np.mean(h)),
                     "median_z_times_h": float(np.median(z * h))})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = json.loads(LABELS.read_text())
    tracks = build_tracks(labels)
    print(f"{len(tracks)} tracks with >= {MIN_TRACK} boxes; lengths {[len(t) for t in tracks]}")
    result = {"tracks": len(tracks), "boxes": int(sum(len(t) for t in tracks)), "folds": {}}
    for fold in FOLDS:
        ranges = predict_ranges(ROOT / f"output/loso_asset/{fold}/weights/best.pt", tracks)
        rows = track_stats(tracks, ranges)
        # tracks with enough size change for the slope to be informative
        informative = [r for r in rows if r["h_ratio"] >= 1.5]
        slopes = np.array([r["loglog_slope"] for r in informative])
        # pooled within-track slope (track-demeaned regression)
        lz, lh = [], []
        for tr in tracks:
            if max(d["h"] for d in tr) / min(d["h"] for d in tr) < 1.5:
                continue
            z = np.log([ranges[(d["frame"], d["cx"], d["cy"])] for d in tr]); h = np.log([d["h"] for d in tr])
            lz += list(z - z.mean()); lh += list(h - h.mean())
        pooled = float(np.dot(lh, lz) / np.dot(lh, lh))
        rng = np.random.default_rng(0)
        boot = [np.median(rng.choice(slopes, len(slopes))) for _ in range(10000)]
        result["folds"][fold] = {"informative_tracks": len(informative), "median_track_slope": float(np.median(slopes)),
                                 "median_slope_95ci": [float(q) for q in np.percentile(boot, [2.5, 97.5])],
                                 "pooled_within_track_slope": pooled,
                                 "median_pearson": float(np.nanmedian([r["pearson_logz_logh"] for r in informative])),
                                 "range_pred_m_p5_p50_p95": [float(q) for q in np.percentile(list(ranges.values()), [5, 50, 95])]}
        if fold == PRIMARY:
            import csv
            with (OUT / "tracks_primary.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            # implied deer height for a range of plausible horizontal fields of view
            W = 724
            fh_px = {hfov: (W / 2) / np.tan(np.radians(hfov / 2)) for hfov in (50, 70, 90, 120)}
            zh = np.array([r["median_z_times_h"] for r in rows])
            result["implied_box_height_m_by_hfov"] = {str(k): [float(q) for q in np.percentile(zh / f, [25, 50, 75])]
                                                      for k, f in fh_px.items()}
        print(fold, json.dumps(result["folds"][fold]))
    agg = np.array([v["pooled_within_track_slope"] for v in result["folds"].values()])
    result["pooled_slope_across_folds"] = {"mean": float(agg.mean()), "min": float(agg.min()), "max": float(agg.max())}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
