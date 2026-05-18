#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flower_task1.data import dataset_summary, save_dataset_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and summarize torchvision Flowers102 official splits.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--no-download", action="store_true", help="Do not download; only inspect existing files.")
    parser.add_argument("--out", type=str, default="data/flowers102_info.json")
    args = parser.parse_args()

    summary = dataset_summary(data_dir=args.data_dir, img_size=args.img_size, download=not args.no_download)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[DONE] Dataset is ready. Summary saved to {out}")


if __name__ == "__main__":
    main()
