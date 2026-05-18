#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flower_task1.data import NUM_CLASSES, make_dataloaders
from flower_task1.models import build_model
from flower_task1.train_utils import AverageMeter, accuracy, resolve_device, save_json


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for images, targets in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        bsz = images.size(0)
        losses.update(loss.item(), bsz)
        top1.update(acc1.item(), bsz)
        top5.update(acc5.item(), bsz)
    return {"loss": losses.avg, "acc1": top1.avg, "acc5": top5.avg}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on val/test split.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=("val", "test"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model_name = ckpt_args.get("model", "resnet18")
    num_classes = int(ckpt_args.get("num_classes", NUM_CLASSES))

    model, _ = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    loaders = make_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, img_size=args.img_size, download=True)
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loaders[args.split], criterion, device)
    metrics.update({"checkpoint": args.checkpoint, "split": args.split, "model": model_name})

    print(metrics)
    if args.out:
        save_json(metrics, args.out)
        print(f"[DONE] Saved metrics to {args.out}")


if __name__ == "__main__":
    main()
