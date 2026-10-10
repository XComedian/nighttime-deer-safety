#!/usr/bin/env python3
"""Regenerate the benchmark's undistorted pinhole views from a local ZOD Frames download.

Usage: python make_views.py --zod /path/to/zod --ann zod_animal_range/annotations.json --out views
Each listed frame's blurred front image (3848x2168 Kannala-Brandt fisheye) is resampled to a 960x540 pinhole view on
the same optical axis with horizontal field of view 54.4 and 70 degrees (fx = 480 / tan(hfov/2), cx = 480, cy = 270).
Requires opencv-python and numpy. ZOD is licensed CC BY-SA 4.0 (Zenseact); derived views keep that license.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zod", type=Path, required=True)
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    frames = sorted({r["frame_id"] for r in json.loads(a.ann.read_text())})
    for hfov in (54.4, 70.0):
        (a.out / f"hfov{hfov:g}").mkdir(parents=True, exist_ok=True)
    for fid in frames:
        d = a.zod / "single_frames" / fid
        calib = json.loads((d / "calibration.json").read_text())["FC"]
        info = json.loads((d / "info.json").read_text())
        img = cv2.imread(str(a.zod / info["camera_frames"]["front_blur"][0]["filepath"]))
        K = np.array(calib["intrinsics"])[:, :3]; D = np.array(calib["distortion"], dtype=np.float64)
        for hfov in (54.4, 70.0):
            f = 480.0 / np.tan(np.radians(hfov / 2))
            P = np.array([[f, 0, 480.0], [0, f, 270.0], [0, 0, 1]])
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, (960, 540), cv2.CV_32FC1)
            cv2.imwrite(str(a.out / f"hfov{hfov:g}" / f"{fid}.jpg"), cv2.remap(img, m1, m2, cv2.INTER_LINEAR))
    print(len(frames), "frames written")


if __name__ == "__main__":
    main()
