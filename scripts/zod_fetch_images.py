#!/usr/bin/env python3
"""Fetch ZOD blurred front images for selected frames only, one archive at a time.

Keep set = every frame with an Animal annotation + a seeded sample of animal-free night/twilight
rural/highway frames (false-alarm negatives). Each 23-25 GB archive is downloaded (resumable),
stream-filtered so only kept frames are written, then deleted before the next archive.
Run with the ZOD CLI venv: .venv_zod/bin/python scripts/zod_fetch_images.py
"""
from __future__ import annotations

import csv
import json
import os
import random
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from zod.cli.download import (APP_KEY, REFRESH_TOKEN, TIMEOUT, DownloadExtractInfo, ResumableDropbox,
                              _download)

URL = os.environ["ZOD_URL"]
ZOD = Path(os.environ.get("ZOD_ROOT", ROOT / "data/zod"))
OUT = Path(__file__).resolve().parents[1] / "output/zod"
ARCHIVES = [("images_blur_000000_049999.tar.gz", 25054466694), ("images_blur_050000_099999.tar.gz", 22946196817)]
N_NEG = 600


def meta(fid: str) -> dict:
    return json.loads((ZOD / "single_frames" / fid / "metadata.json").read_text())


def keep_set() -> set[str]:
    animal = {r["frame"] for r in csv.DictReader((OUT / "animal_annotations.csv").open())}
    fids = sorted(p.name for p in (ZOD / "single_frames").iterdir())
    with ProcessPoolExecutor(32) as ex:
        metas = list(ex.map(meta, fids, chunksize=1000))
    pool = [m["frame_id"] for m in metas if m["frame_id"] not in animal and m["time_of_day"] != "day"
            and m["road_type"] in ("smaller-rural", "arterial-rural", "highway")]
    negatives = random.Random(0).sample(pool, min(N_NEG, len(pool)))
    (OUT / "negative_frames.json").write_text(json.dumps(sorted(negatives)))
    print(f"keep: {len(animal)} animal frames + {len(negatives)} negatives (pool {len(pool)})", flush=True)
    return animal | set(negatives)


def main() -> None:
    keep = keep_set()
    dbx = ResumableDropbox(app_key=APP_KEY, oauth2_refresh_token=REFRESH_TOKEN, timeout=TIMEOUT)
    dl_dir = ZOD / "downloads"
    dl_dir.mkdir(exist_ok=True)
    for name, size in ARCHIVES:
        path = dl_dir / name
        info = DownloadExtractInfo(url=URL, file_path=f"/single_frames/{name}", extract_dir=str(ZOD),
                                   dl_dir=str(dl_dir), rm=True, dry_run=False, size=size, content_hash="",
                                   extract=False, extract_already_downloaded=False)
        if not (path.exists() and path.stat().st_size == size):
            _download(str(path), dbx, info)
        print(f"downloaded {name}", flush=True)
        kept = 0
        with tarfile.open(path, "r|gz") as tar:          # streaming: no index, one pass
            for member in tar:
                parts = Path(member.name).parts           # single_frames/<fid>/camera_front_blur/<file>.jpg
                if member.isfile() and len(parts) >= 2 and parts[1] in keep:
                    tar.extract(member, ZOD)
                    kept += 1
        path.unlink()
        print(f"extracted {kept} images from {name}; archive deleted", flush=True)


if __name__ == "__main__":
    main()
