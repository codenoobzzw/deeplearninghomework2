from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import CLASS_NAMES, StanfordBackgroundDataset, denormalize
from model import UNet

PALETTE = np.array(
    [
        [135, 206, 235],
        [34, 139, 34],
        [128, 128, 128],
        [124, 252, 0],
        [30, 144, 255],
        [178, 34, 34],
        [139, 137, 137],
        [255, 215, 0],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)


def colorize(mask: np.ndarray) -> np.ndarray:
    safe = mask.copy()
    safe[(safe < 0) | (safe >= len(PALETTE))] = len(PALETTE) - 1
    return PALETTE[safe]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize U-Net predictions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--out-dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt.get("config", {})
    image_size = int(config.get("image_size", 256))
    base_channels = int(config.get("base_channels", 32))
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    ds = StanfordBackgroundDataset(args.data_dir, split=args.split, image_size=image_size, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    model = UNet(in_channels=3, num_classes=len(CLASS_NAMES), base_channels=base_channels)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if saved >= args.num_samples:
                break
            image = batch["image"].to(device)
            mask = batch["mask"][0].cpu().numpy()
            logits = model(image)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int64)
            img_np = denormalize(batch["image"][0]).permute(1, 2, 0).cpu().numpy()
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np)
            axes[0].set_title("image")
            axes[1].imshow(colorize(mask))
            axes[1].set_title("ground truth")
            axes[2].imshow(colorize(pred))
            axes[2].set_title("prediction")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            out = args.out_dir / f"prediction_{i:03d}.jpg"
            fig.savefig(out, dpi=160)
            plt.close(fig)
            saved += 1
    print(f"Saved {saved} prediction visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()
