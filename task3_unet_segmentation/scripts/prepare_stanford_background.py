from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_KAGGLE_SLUGS = ["ipythonx/stanford-background-dataset", "balraj98/stanford-background-dataset"]
CLASS_NAMES = ["sky", "tree", "road", "grass", "water", "building", "mountain", "foreground"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def run_cmd(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_kaggle(slugs: list[str], download_dir: Path) -> Path:
    if shutil.which("kaggle") is None:
        raise RuntimeError("kaggle CLI not found. Configure Kaggle or use --source-dir.")
    download_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for slug in slugs:
        try:
            run_cmd(["kaggle", "datasets", "download", "-d", slug, "-p", str(download_dir), "--unzip"])
            return download_dir
        except Exception as exc:
            last_error = exc
            print(f"Download failed for {slug}: {exc}")
    raise RuntimeError(f"All Kaggle downloads failed: {last_error}")


def find_dir(root: Path, name: str) -> Path | None:
    candidates = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == name.lower()]
    if candidates:
        candidates.sort(key=lambda p: len(str(p)))
        return candidates[0]
    return None


def find_images_dir(root: Path) -> Path:
    d = find_dir(root, "images")
    if d is not None:
        return d
    image_files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]
    if not image_files:
        raise RuntimeError(f"No images found under {root}")
    counts: dict[Path, int] = {}
    for p in image_files:
        counts[p.parent] = counts.get(p.parent, 0) + 1
    return max(counts, key=counts.get)


def find_label_file(image: Path, root: Path) -> Path | None:
    stem = image.stem
    candidates = [
        root / "labels" / f"{stem}.regions.txt",
        root / "labels" / f"{stem}.regions",
        root / "labels" / f"{stem}.png",
        root / "label" / f"{stem}.regions.txt",
        root / "label" / f"{stem}.png",
        root / "labels_colored" / f"{stem}.png",
        root / "annotations" / f"{stem}.regions.txt",
        root / "annotations" / f"{stem}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = list(root.rglob(f"{stem}.regions.txt")) + list(root.rglob(f"{stem}.png"))
    if matches:
        matches.sort(key=lambda p: len(str(p)))
        return matches[0]
    return None


def load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".txt" or path.name.endswith(".regions"):
        arr = np.loadtxt(path, dtype=np.int64)
    else:
        img = Image.open(path)
        arr = np.array(img)
        if arr.ndim == 3:
            # RGB colored labels are less reliable because palettes differ. Convert unique colors to ids.
            colors = arr.reshape(-1, arr.shape[-1])
            unique = np.unique(colors, axis=0)
            mapping = {tuple(color.tolist()): i for i, color in enumerate(unique[: len(CLASS_NAMES)])}
            out = np.zeros(arr.shape[:2], dtype=np.int64)
            for color, idx in mapping.items():
                mask = np.all(arr == np.array(color, dtype=arr.dtype), axis=-1)
                out[mask] = idx
            arr = out
        else:
            arr = arr.astype(np.int64)
    arr = np.asarray(arr, dtype=np.int64)
    valid = arr[arr != 255]
    if valid.size > 0 and valid.min() >= 1 and valid.max() <= len(CLASS_NAMES):
        arr = arr - 1
    arr[(arr < 0) | (arr >= len(CLASS_NAMES))] = 255
    return arr.astype(np.uint8)


def split_pairs(pairs: list[tuple[Path, Path]], seed: int) -> dict[str, list[tuple[Path, Path]]]:
    rng = random.Random(seed)
    pairs = pairs[:]
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }


def write_processed_dataset(source_dir: Path, out_dir: Path, seed: int) -> None:
    images_dir = find_images_dir(source_dir)
    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file())
    pairs: list[tuple[Path, Path]] = []
    missing = 0
    for img in images:
        label = find_label_file(img, source_dir)
        if label is None:
            missing += 1
            continue
        pairs.append((img, label))
    if not pairs:
        raise RuntimeError("No image/mask pairs found. Expected labels/*.regions.txt or mask png files.")
    print(f"Found {len(pairs)} image/mask pairs; {missing} images without labels skipped.")

    processed_img = out_dir / "processed" / "images"
    processed_mask = out_dir / "processed" / "masks"
    processed_img.mkdir(parents=True, exist_ok=True)
    processed_mask.mkdir(parents=True, exist_ok=True)

    processed_pairs: list[tuple[Path, Path]] = []
    for idx, (img, label) in enumerate(pairs):
        dst_img = processed_img / f"{idx:04d}_{img.stem}.jpg"
        dst_mask = processed_mask / f"{idx:04d}_{img.stem}.png"
        image = Image.open(img).convert("RGB")
        image.save(dst_img, quality=95)
        mask = load_mask(label)
        Image.fromarray(mask, mode="L").save(dst_mask)
        processed_pairs.append((dst_img, dst_mask))

    splits = split_pairs(processed_pairs, seed)
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, split_pairs_ in splits.items():
        with (split_dir / f"{split}.txt").open("w", encoding="utf-8") as f:
            for img, mask in split_pairs_:
                f.write(f"{img.resolve()}\t{mask.resolve()}\n")

    info = {
        "source_dir": str(source_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "ignore_index": 255,
        "splits": {k: len(v) for k, v in splits.items()},
        "image_dir_detected": str(images_dir.resolve()),
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stanford Background Dataset for U-Net training.")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--kaggle", action="store_true")
    parser.add_argument("--kaggle-slug", action="append", default=None, help="May be passed multiple times.")
    parser.add_argument("--download-dir", type=Path, default=Path("downloads/stanford_background_raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/stanford_background"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.kaggle:
        slugs = args.kaggle_slug or DEFAULT_KAGGLE_SLUGS
        source = download_kaggle(slugs, args.download_dir)
    elif args.source_dir:
        source = args.source_dir
    else:
        parser.error("Pass --kaggle or --source-dir")
        return

    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_processed_dataset(source, args.out_dir, args.seed)
    print("Done. Next: python scripts/train.py --data-dir", args.out_dir)


if __name__ == "__main__":
    main()
