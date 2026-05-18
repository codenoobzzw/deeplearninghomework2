#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flower_task1.data import NUM_CLASSES, make_dataloaders
from flower_task1.models import SUPPORTED_MODELS, build_model, count_parameters, parameter_groups
from flower_task1.train_utils import (
    AverageMeter,
    accuracy,
    describe_device,
    get_learning_rates,
    resolve_device,
    save_checkpoint,
    save_history_csv,
    save_json,
    set_seed,
)


class Tracker:
    def __init__(self, kind: str, args: argparse.Namespace, run_dir: Path, config: Dict) -> None:
        self.kind = kind.lower()
        self.run = None
        self.enabled = False
        if self.kind == "none":
            return
        try:
            if self.kind == "wandb":
                os.environ.setdefault("WANDB_MODE", args.wandb_mode)
                import wandb

                self.run = wandb.init(
                    project=args.wandb_project,
                    name=args.exp_name,
                    dir=str(run_dir),
                    config=config,
                    reinit=True,
                )
                self.enabled = True
            elif self.kind == "swanlab":
                import swanlab

                self.run = swanlab.init(
                    project=args.swanlab_project,
                    experiment_name=args.exp_name,
                    config=config,
                    logdir=str(run_dir / "swanlog"),
                )
                self.enabled = True
            else:
                raise ValueError(f"Unknown tracker: {self.kind}")
        except Exception as exc:  # noqa: BLE001 - local logging should still work even if tracker is unavailable.
            print(f"[WARN] Failed to initialize tracker '{self.kind}': {exc}. Continue with local logs only.")
            self.enabled = False

    def log(self, metrics: Dict, step: int | None = None) -> None:
        if not self.enabled:
            return
        try:
            if self.kind == "wandb":
                import wandb

                wandb.log(metrics, step=step)
            elif self.kind == "swanlab":
                import swanlab

                swanlab.log(metrics, step=step)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Tracker log failed: {exc}")

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            if self.kind == "wandb":
                import wandb

                wandb.finish()
            elif self.kind == "swanlab":
                import swanlab

                swanlab.finish()
        except Exception:
            pass


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    grad_clip_norm: float,
) -> Dict[str, float]:
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        bsz = images.size(0)
        losses.update(loss.item(), bsz)
        top1.update(acc1.item(), bsz)
        top5.update(acc5.item(), bsz)
        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc1=f"{top1.avg:.2f}")

    return {"loss": losses.avg, "acc1": top1.avg, "acc5": top5.avg}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split: str = "val",
    use_amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    pbar = tqdm(loader, desc=f"eval {split}", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with amp_context(device, use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        bsz = images.size(0)
        losses.update(loss.item(), bsz)
        top1.update(acc1.item(), bsz)
        top5.update(acc5.item(), bsz)
        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc1=f"{top1.avg:.2f}")
    return {"loss": losses.avg, "acc1": top1.avg, "acc5": top5.avg}


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    groups = parameter_groups(model, lr_backbone=args.lr_backbone, lr_head=args.lr_head, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if args.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)
    if args.scheduler == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1: fine-tune ResNet/attention ResNet on Oxford Flowers102.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for Flowers102 data.")
    parser.add_argument("--output-dir", type=str, default="runs", help="Directory to store run outputs.")
    parser.add_argument("--exp-name", type=str, default="", help="Experiment name. Auto-generated if empty.")
    parser.add_argument("--model", type=str, default="resnet18", choices=SUPPORTED_MODELS)
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument("--pretrained", dest="pretrained", action="store_true", help="Use ImageNet pretrained backbone.")
    pretrained_group.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Randomly initialize the whole network.")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Learning rate for new fc classifier head.")
    parser.add_argument("--lr-backbone", type=float, default=1e-4, help="Learning rate for pretrained/non-fc layers.")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=("adamw", "sgd"))
    parser.add_argument("--scheduler", type=str, default="cosine", choices=("cosine", "step", "none"))
    parser.add_argument("--step-size", type=int, default=15)
    parser.add_argument("--step-gamma", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    download_group = parser.add_mutually_exclusive_group()
    download_group.add_argument("--download", dest="download", action="store_true")
    download_group.add_argument("--no-download", dest="download", action="store_false")
    parser.set_defaults(download=True)
    parser.add_argument("--tracker", type=str, default="none", choices=("none", "wandb", "swanlab"))
    parser.add_argument("--wandb-project", type=str, default="dl-hw2-task1-flowers102")
    parser.add_argument("--wandb-mode", type=str, default="offline", choices=("online", "offline", "disabled"))
    parser.add_argument("--swanlab-project", type=str, default="dl-hw2-task1-flowers102")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience in epochs. 0 disables it.")
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--test-at-end", dest="test_at_end", action="store_true")
    test_group.add_argument("--no-test-at-end", dest="test_at_end", action="store_false")
    parser.set_defaults(test_at_end=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.exp_name:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        init_tag = "pretrained" if args.pretrained else "random"
        args.exp_name = f"{stamp}_{args.model}_{init_tag}_h{args.lr_head:g}_b{args.lr_backbone:g}"

    run_dir = Path(args.output_dir) / args.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()

    device = resolve_device(args.device)
    device_desc = describe_device(device)
    print(f"[INFO] Device: {device_desc}")
    if args.amp and device.type != "cuda":
        print("[WARN] --amp was requested but CUDA is unavailable. AMP disabled.")
        args.amp = False

    loaders = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        download=args.download,
    )
    config["num_train_images"] = len(loaders["train"].dataset)
    config["num_val_images"] = len(loaders["val"].dataset)
    config["num_test_images"] = len(loaders["test"].dataset)
    config["device"] = device_desc

    model, load_report = build_model(args.model, num_classes=args.num_classes, pretrained=args.pretrained)
    model.to(device)
    config["model_load_report"] = load_report
    config["parameters"] = count_parameters(model)
    save_json(config, run_dir / "config.json")
    print(f"[INFO] Model: {args.model}, pretrained={args.pretrained}, params={config['parameters']}")
    print(f"[INFO] Run dir: {run_dir}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    tracker = Tracker(args.tracker, args, run_dir, config=config)

    history = []
    best_val_acc1 = -1.0
    best_epoch = -1
    epochs_without_improve = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_amp=args.amp,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device, split="val", use_amp=args.amp)
        if scheduler is not None:
            scheduler.step()

        global_step += len(loaders["train"])
        lrs = get_learning_rates(optimizer)
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc1": train_metrics["acc1"],
            "train_acc5": train_metrics["acc5"],
            "val_loss": val_metrics["loss"],
            "val_acc1": val_metrics["acc1"],
            "val_acc5": val_metrics["acc5"],
            **lrs,
        }
        history.append(record)
        save_json({"history": history}, run_dir / "history.json")
        save_history_csv(history, run_dir / "history.csv")

        tracker.log(record, step=global_step)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={record['train_loss']:.4f} train_acc1={record['train_acc1']:.2f} | "
            f"val_loss={record['val_loss']:.4f} val_acc1={record['val_acc1']:.2f} val_acc5={record['val_acc5']:.2f}"
        )

        improved = val_metrics["acc1"] > best_val_acc1
        if improved:
            best_val_acc1 = float(val_metrics["acc1"])
            best_epoch = epoch
            epochs_without_improve = 0
            save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_acc1,
                vars(args),
                extra={"config": config, "history": history},
            )
            print(f"[INFO] New best checkpoint saved: epoch={epoch}, val_acc1={best_val_acc1:.2f}")
        else:
            epochs_without_improve += 1

        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_acc1,
            vars(args),
            extra={"config": config, "history": history},
        )

        metrics_best = {
            "exp_name": args.exp_name,
            "model": args.model,
            "pretrained": args.pretrained,
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "best_val_acc1": best_val_acc1,
            "best_val_acc5": max([r["val_acc5"] for r in history], default=0.0),
            "lr_head": args.lr_head,
            "lr_backbone": args.lr_backbone,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "label_smoothing": args.label_smoothing,
            "run_dir": str(run_dir),
            "best_checkpoint": str(run_dir / "best.pt"),
        }
        save_json(metrics_best, run_dir / "metrics_best.json")

        if args.patience > 0 and epochs_without_improve >= args.patience:
            print(f"[INFO] Early stopping triggered after {epochs_without_improve} epochs without improvement.")
            break

    final_metrics = {
        "exp_name": args.exp_name,
        "model": args.model,
        "pretrained": args.pretrained,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_acc1": best_val_acc1,
        "lr_head": args.lr_head,
        "lr_backbone": args.lr_backbone,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "best.pt"),
    }

    if args.test_at_end and (run_dir / "best.pt").exists():
        print("[INFO] Loading best checkpoint for final test evaluation...")
        checkpoint = torch.load(run_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_metrics = evaluate(model, loaders["test"], criterion, device, split="test", use_amp=args.amp)
        final_metrics.update(
            {
                "test_loss": test_metrics["loss"],
                "test_acc1": test_metrics["acc1"],
                "test_acc5": test_metrics["acc5"],
            }
        )
        tracker.log({f"test_{k}": v for k, v in test_metrics.items()}, step=global_step + 1)
        print(f"[INFO] Test: loss={test_metrics['loss']:.4f}, acc1={test_metrics['acc1']:.2f}, acc5={test_metrics['acc5']:.2f}")

    save_json(final_metrics, run_dir / "metrics_best.json")
    tracker.finish()
    print(f"[DONE] Finished run '{args.exp_name}'. Best val acc1={best_val_acc1:.2f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
