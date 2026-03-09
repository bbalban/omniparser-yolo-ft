#!/usr/bin/env python3
"""Capture XFCE screenshots at fixed intervals."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def run_scrot(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scrot", str(output_path)]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture screenshots from XFCE.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/screenshots"),
        help="Output directory for PNG screenshots.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/manifest.jsonl"),
        help="Manifest JSONL path.",
    )
    parser.add_argument("--count", type=int, default=10, help="Number of screenshots.")
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=2.5,
        help="Seconds between captures.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="shot",
        help="Filename prefix for screenshots.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    print(f"[capture] Writing {args.count} screenshots to {args.out_dir}")
    print("[capture] Ensure the XFCE workflow is active (dialogs, menus, save UI, etc.).")

    with args.manifest.open("a", encoding="utf-8") as manifest_f:
        for idx in range(1, args.count + 1):
            filename = f"{args.prefix}_{idx:04d}.png"
            path = args.out_dir / filename
            run_scrot(path)
            item = {
                "image": filename,
                "abs_path": str(path.resolve()),
                "captured_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_f.write(json.dumps(item) + "\n")
            print(f"[capture] {idx:02d}/{args.count}: {filename}")
            if idx < args.count:
                time.sleep(args.interval_sec)

    print("[capture] done")


if __name__ == "__main__":
    main()
