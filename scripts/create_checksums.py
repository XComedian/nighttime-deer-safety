#!/usr/bin/env python3
"""Create a SHA-256 manifest for a dataset or checkpoint archive."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("SHA256SUMS"))
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.resolve() != output
    )
    lines = [f"{sha256(path)}  {path.relative_to(root)}" for path in paths]
    output.write_text("\n".join(lines) + "\n")
    print(f"wrote {output} with {len(paths)} entries")


if __name__ == "__main__":
    main()
