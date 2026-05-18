from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CLASS_NAMES, IGNORE_INDEX, StanfordBackgroundDataset
from metrics import SegmentationMeter
from model import UNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt.get("config", {})
    image_size = args.image_size or int(config.get("image_size", 256))
    base_channels = args.base_channels or int(config.get("base_channels", 32))
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    ds = StanfordBackgroundDataset(args.data_dir, split=args.split, image_size=image_size, augment=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = UNet(in_channels=3, num_classes=len(CLASS_NAMES), base_channels=base_channels)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    meter = SegmentationMeter(num_classes=len(CLASS_NAMES), ignore_index=IGNORE_INDEX)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate {args.split}"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            meter.update(logits, masks)
    metrics = meter.compute()
    metrics["split"] = args.split
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["class_names"] = CLASS_NAMES
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
