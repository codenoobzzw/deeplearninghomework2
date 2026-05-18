from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CLASS_NAMES, IGNORE_INDEX, StanfordBackgroundDataset
from losses import build_loss
from metrics import SegmentationMeter
from model import UNet, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Tracker:
    def __init__(self, name: str, project: str, run_name: str, output_dir: Path, wandb_mode: str = "offline") -> None:
        self.name = name
        self.backend: Any = None
        if name == "wandb":
            try:
                import wandb

                os.environ["WANDB_MODE"] = wandb_mode
                self.backend = wandb.init(project=project, name=run_name, dir=str(output_dir), mode=wandb_mode)
            except Exception as exc:
                print(f"wandb init failed, continuing without tracker: {exc}")
                self.name = "none"
        elif name == "swanlab":
            try:
                import swanlab

                self.backend = swanlab.init(project=project, experiment_name=run_name, logdir=str(output_dir))
            except Exception as exc:
                print(f"swanlab init failed, continuing without tracker: {exc}")
                self.name = "none"

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self.name == "wandb" and self.backend is not None:
            import wandb

            wandb.log(metrics, step=step)
        elif self.name == "swanlab" and self.backend is not None:
            import swanlab

            swanlab.log(metrics, step=step)

    def finish(self) -> None:
        if self.name == "wandb" and self.backend is not None:
            import wandb

            wandb.finish()
        elif self.name == "swanlab" and self.backend is not None:
            try:
                import swanlab

                swanlab.finish()
            except Exception:
                pass


def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_classes: int,
    amp: bool,
    scaler: GradScaler | None,
    desc: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_samples = 0
    meter = SegmentationMeter(num_classes=num_classes, ignore_index=IGNORE_INDEX)

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        bs = images.size(0)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with autocast(enabled=amp):
                logits = model(images)
                loss = criterion(logits, masks)
            if is_train:
                assert optimizer is not None
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().cpu()) * bs
        n_samples += bs
        meter.update(logits.detach(), masks.detach())
        metrics = meter.compute()
        pbar.set_postfix(loss=total_loss / max(n_samples, 1), miou=metrics["miou"])

    out = meter.compute()
    return {
        "loss": total_loss / max(n_samples, 1),
        "miou": float(out["miou"]),
        "pixel_acc": float(out["pixel_acc"]),
        "mean_acc": float(out["mean_acc"]),
    }


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, Any], config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "class_names": CLASS_NAMES,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a handwritten U-Net on Stanford Background Dataset.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--loss", choices=["ce", "dice", "ce_dice"], default="ce")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--tracker", choices=["none", "wandb", "swanlab"], default="none")
    parser.add_argument("--tracker-project", default="dl-hw2-task3-unet")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config["num_classes"] = len(CLASS_NAMES)
    config["class_names"] = CLASS_NAMES
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    train_ds = StanfordBackgroundDataset(args.data_dir, split="train", image_size=args.image_size, augment=True)
    val_ds = StanfordBackgroundDataset(args.data_dir, split="val", image_size=args.image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = UNet(in_channels=3, num_classes=len(CLASS_NAMES), base_channels=args.base_channels).to(device)
    criterion = build_loss(args.loss, num_classes=len(CLASS_NAMES), ignore_index=IGNORE_INDEX, ce_weight=args.ce_weight, dice_weight=args.dice_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = GradScaler(enabled=args.amp)
    tracker = Tracker(args.tracker, args.tracker_project, args.output_dir.name, args.output_dir, args.wandb_mode)

    print(f"Device: {device}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Train images: {len(train_ds)}, Val images: {len(val_ds)}")
    print(f"Loss: {args.loss}")

    history: list[dict[str, Any]] = []
    best_miou = -1.0
    best_metrics: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, len(CLASS_NAMES), args.amp, scaler, f"train {epoch}/{args.epochs}")
        val_metrics = run_one_epoch(model, val_loader, criterion, None, device, len(CLASS_NAMES), args.amp, None, f"val {epoch}/{args.epochs}")
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_miou": train_metrics["miou"],
            "train_pixel_acc": train_metrics["pixel_acc"],
            "train_mean_acc": train_metrics["mean_acc"],
            "val_loss": val_metrics["loss"],
            "val_miou": val_metrics["miou"],
            "val_pixel_acc": val_metrics["pixel_acc"],
            "val_mean_acc": val_metrics["mean_acc"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        tracker.log(row, step=epoch)
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_miou={row['val_miou']:.4f} val_pixel_acc={row['val_pixel_acc']:.4f}"
        )
        if val_metrics["miou"] > best_miou:
            best_miou = float(val_metrics["miou"])
            best_metrics = row.copy()
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, best_metrics, config)
            (args.output_dir / "metrics_best.json").write_text(json.dumps(best_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, row, config)

    tracker.finish()
    print("Best metrics:")
    print(json.dumps(best_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
