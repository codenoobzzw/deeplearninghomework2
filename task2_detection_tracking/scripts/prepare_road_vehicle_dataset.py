from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_KAGGLE_SLUG = "ashfakyeafi/road-vehicle-images-dataset"


def run_cmd(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_kaggle(slug: str, download_dir: Path) -> Path:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "kaggle CLI not found. Install kaggle and configure ~/.kaggle/kaggle.json, "
            "or pass --source-dir with a manually uploaded dataset."
        )
    download_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(["kaggle", "datasets", "download", "-d", slug, "-p", str(download_dir), "--unzip"])
    return download_dir


def safe_load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def find_data_yaml(root: Path) -> Path | None:
    candidates = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))
    scored: list[tuple[int, Path]] = []
    for p in candidates:
        cfg = safe_load_yaml(p)
        if not cfg:
            continue
        keys = set(cfg.keys())
        score = 0
        if "train" in keys:
            score += 2
        if "val" in keys or "valid" in keys:
            score += 2
        if "names" in keys:
            score += 2
        if p.name.lower() in {"data.yaml", "dataset.yaml"}:
            score += 1
        if score >= 4:
            scored.append((score, p))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], len(str(x[1]))))
    return scored[0][1]


def normalize_names(names: Any, n_classes: int | None = None) -> dict[int, str]:
    if isinstance(names, dict):
        out = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        out = {i: str(v) for i, v in enumerate(names)}
    else:
        out = {}
    if not out and n_classes is not None:
        out = {i: f"class_{i}" for i in range(n_classes)}
    return dict(sorted(out.items()))


def copy_existing_yaml(yaml_path: Path, out_dir: Path) -> dict[str, Any]:
    cfg = safe_load_yaml(yaml_path)
    if not cfg:
        raise RuntimeError(f"Could not parse YAML: {yaml_path}")

    base = yaml_path.parent.resolve()
    original_path = cfg.get("path")
    if original_path:
        p = Path(str(original_path))
        cfg["path"] = str((base / p).resolve() if not p.is_absolute() else p.resolve())
    else:
        cfg["path"] = str(base)

    if "valid" in cfg and "val" not in cfg:
        cfg["val"] = cfg.pop("valid")
    if "names" in cfg:
        cfg["names"] = normalize_names(cfg["names"])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = out_dir / "data.yaml"
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"Found existing YOLO YAML: {yaml_path}")
    print(f"Wrote normalized YAML: {out_yaml}")
    return cfg


def find_image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file())


def find_label_for_image(img: Path, root: Path) -> Path | None:
    stem = img.stem + ".txt"
    candidates = [
        img.with_suffix(".txt"),
        img.parent.parent / "labels" / stem,
        root / "labels" / stem,
    ]
    parts = list(img.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            new_parts = parts[:i] + ["labels"] + parts[i + 1 :]
            candidates.append(Path(*new_parts).with_suffix(".txt"))
    for c in candidates:
        if c.exists():
            return c
    matches = list(root.rglob(stem))
    return matches[0] if matches else None


def infer_class_count(label_files: list[Path]) -> int:
    max_id = -1
    for label in label_files:
        try:
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if parts:
                    max_id = max(max_id, int(float(parts[0])))
        except Exception:
            continue
    return max_id + 1 if max_id >= 0 else 1


def find_classes(root: Path, n_classes: int) -> dict[int, str]:
    for name in ["classes.txt", "obj.names", "data.names"]:
        matches = list(root.rglob(name))
        if matches:
            lines = [x.strip() for x in matches[0].read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
            if lines:
                return normalize_names(lines)
    return normalize_names(None, n_classes)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        try:
            os.symlink(src.resolve(), dst)
        except Exception:
            shutil.copy2(src, dst)


def split_items(items: list[tuple[Path, Path]], seed: int) -> dict[str, list[tuple[Path, Path]]]:
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


def get_split_name(path: Path) -> str | None:
    lowered = [p.lower() for p in path.parts]
    if "train" in lowered or "training" in lowered:
        return "train"
    if "valid" in lowered or "val" in lowered or "validation" in lowered:
        return "val"
    if "test" in lowered or "testing" in lowered:
        return "test"
    return None


def build_dataset_from_files(source_dir: Path, out_dir: Path, link_mode: str, seed: int) -> dict[str, Any]:
    images = find_image_files(source_dir)
    pairs: list[tuple[Path, Path]] = []
    missing = 0
    for img in images:
        label = find_label_for_image(img, source_dir)
        if label is None:
            missing += 1
            continue
        pairs.append((img, label))
    if not pairs:
        raise RuntimeError(f"No image/YOLO-label pairs found under {source_dir}")
    print(f"Found {len(pairs)} image/label pairs; {missing} images without labels were skipped.")

    by_split: dict[str, list[tuple[Path, Path]]] = {"train": [], "val": [], "test": []}
    for img, label in pairs:
        split = get_split_name(img.relative_to(source_dir)) or get_split_name(label.relative_to(source_dir))
        if split:
            by_split[split].append((img, label))
    if len(by_split["train"]) == 0 or len(by_split["val"]) == 0:
        by_split = split_items(pairs, seed)

    for split, split_pairs in by_split.items():
        for idx, (img, label) in enumerate(split_pairs):
            suffix = img.suffix.lower()
            dst_img = out_dir / split / "images" / f"{img.stem}_{idx:06d}{suffix}"
            dst_lab = out_dir / split / "labels" / f"{img.stem}_{idx:06d}.txt"
            link_or_copy(img, dst_img, link_mode)
            link_or_copy(label, dst_lab, link_mode)

    label_files = [label for _, label in pairs]
    n_classes = infer_class_count(label_files)
    names = find_classes(source_dir, n_classes)
    cfg: dict[str, Any] = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images" if by_split["test"] else "val/images",
        "names": names,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    info = {
        "source_dir": str(source_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "splits": {k: len(v) for k, v in by_split.items()},
        "names": names,
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Road Vehicle Images Dataset for YOLOv8.")
    parser.add_argument("--source-dir", type=Path, default=None, help="Existing dataset directory.")
    parser.add_argument("--kaggle", action="store_true", help="Download from Kaggle with kaggle CLI.")
    parser.add_argument("--kaggle-slug", default=DEFAULT_KAGGLE_SLUG)
    parser.add_argument("--download-dir", type=Path, default=Path("downloads/road_vehicle_raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/road_vehicle_yolo"))
    parser.add_argument("--link-mode", choices=["auto", "symlink", "copy"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.kaggle:
        source = download_kaggle(args.kaggle_slug, args.download_dir)
    elif args.source_dir:
        source = args.source_dir
    else:
        parser.error("Pass --kaggle or --source-dir")
        return

    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    yaml_path = find_data_yaml(source)
    if yaml_path is not None:
        cfg = copy_existing_yaml(yaml_path, args.out_dir)
        (args.out_dir / "dataset_info.json").write_text(
            json.dumps({"source_dir": str(source), "data_yaml": str(yaml_path), "config": cfg}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        build_dataset_from_files(source, args.out_dir, args.link_mode, args.seed)

    print("Done. Use:")
    print(f"  python scripts/train_yolo.py --data {args.out_dir / 'data.yaml'} --model yolov8n.pt --epochs 1")


if __name__ == "__main__":
    main()
