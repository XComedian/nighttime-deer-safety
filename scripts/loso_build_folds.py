#!/usr/bin/env python3
"""Build leave-one-scene-out folds from converted LOSO scenes.

For each scene tag, one fold: the held-out scene is the TEST scene and is never seen
during training -- not as training data, and not as the validation set.

Earlier versions pointed `val:` at the held-out scene, which meant checkpoint
selection and early stopping were driven by the test scene (test-set leakage). Here
validation is a strided subset carved out of the TRAINING scenes only: every
VAL_STRIDE-th frame of each training scene goes to val.txt, the rest to train.txt.
That val split is correlated with training (same scenes) and is used solely to pick a
checkpoint and stop early; the test scene is evaluated exactly once, afterwards, by
loso_eval.py.

Usage: loso_build_folds.py c7 c8 c9a c9b c9c c10
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "data" / "loso" / "scenes"
FOLDS = ROOT / "data" / "loso" / "folds"
VAL_STRIDE = 8   # every 8th training frame -> inner validation


def main() -> None:
    tags = sys.argv[1:]
    if len(tags) < 2:
        raise SystemExit("need >=2 scene tags")
    for t in tags:
        if not (SCENES / t / "images").is_dir():
            raise SystemExit(f"missing converted scene: {SCENES / t}")

    for held in tags:
        train_tags = [t for t in tags if t != held]
        fold = FOLDS / held
        fold.mkdir(parents=True, exist_ok=True)

        # Per-fold symlink tree so each fold gets its OWN labels.cache.
        # Ultralytics derives the cache path lexically as
        #   Path(label_files[0]).parent.with_suffix(".cache")
        # so folds pointing at the shared scenes/<tag>/labels/ all raced on one
        # cache file -- that killed one fold per parallel run (c9b, then c8).
        # Linking images/ and labels/ under a per-fold directory moves the cache
        # to folds/<held>/link/<tag>/labels.cache without duplicating any data.
        link_root = fold / "link"
        train_imgs, val_imgs = [], []
        for t in train_tags:
            ldir = link_root / t
            ldir.mkdir(parents=True, exist_ok=True)
            for sub in ("images", "labels"):
                dst, src = ldir / sub, SCENES / t / sub
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(src, target_is_directory=True)
            imgs = sorted((SCENES / t / "images").glob("*.jpg"))
            for i, p in enumerate(imgs):
                linked = str(ldir / "images" / p.name)
                (val_imgs if i % VAL_STRIDE == 0 else train_imgs).append(linked)

        (fold / "train.txt").write_text("\n".join(train_imgs) + "\n")
        (fold / "val.txt").write_text("\n".join(val_imgs) + "\n")

        # Test scene deliberately absent from the yaml: it is never loaded in training.
        (fold / "data.yaml").write_text(
            f"path: {ROOT / 'data' / 'loso'}\n"
            f"train: {fold / 'train.txt'}\n"
            f"val: {fold / 'val.txt'}\n"
            "nc: 1\nnames: [deer]\n"
        )

        merged: dict[str, list] = {}
        for t in train_tags:
            merged.update(json.loads((SCENES / t / "distances.json").read_text()))
        (fold / "train_distances.json").write_text(json.dumps(merged))
        (fold / "held.txt").write_text(held + "\n")
        print(f"fold {held}: train {train_tags} | {len(train_imgs)} train + "
              f"{len(val_imgs)} inner-val frames; test scene {held} unseen")


if __name__ == "__main__":
    main()
