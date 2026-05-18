from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_split_dir(data_yaml: Path, split: str) -> Path:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(cfg.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = data_yaml.parent / base
    rel = cfg.get(split) or cfg.get("val")
    p = Path(str(rel))
    return p if p.is_absolute() else base / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a simple demo video from dataset images for code testing only.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", type=Path, default=Path("outputs/demo_from_images.mp4"))
    parser.add_argument("--seconds", type=int, default=12)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    img_dir = resolve_split_dir(args.data, args.split)
    images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise RuntimeError(f"No images found in {img_dir}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    n_frames = args.seconds * args.fps
    for i in range(n_frames):
        img = cv2.imread(str(images[i % len(images)]))
        if img is None:
            continue
        img = cv2.resize(img, (args.width, args.height))
        cv2.putText(img, "DEMO VIDEO FROM STATIC IMAGES - NOT FOR FINAL REPORT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(img)
    writer.release()
    print(f"Wrote demo video: {args.out}")
    print("Use this only for testing the code path. The final report should use a real 10-30s video.")


if __name__ == "__main__":
    main()
