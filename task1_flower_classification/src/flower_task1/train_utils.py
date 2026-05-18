from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1,)) -> List[torch.Tensor]:
    """Compute top-k accuracy percentages for classification."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res: List[torch.Tensor] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def as_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(x) for x in obj]
    return obj


def save_json(data: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def save_history_csv(history: List[Dict], path: str | Path) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in history for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_acc1: float,
    args: Dict,
    extra: Dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "best_val_acc1": best_val_acc1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "args": as_jsonable(args),
    }
    if extra:
        state.update(as_jsonable(extra))
    torch.save(state, path)


def get_learning_rates(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for i, group in enumerate(optimizer.param_groups):
        name = str(group.get("group_name", f"group_{i}"))
        result[f"lr_{name}"] = float(group["lr"])
    return result


def resolve_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def describe_device(device: torch.device) -> Dict[str, object]:
    desc: Dict[str, object] = {"device": str(device), "cuda_available": torch.cuda.is_available()}
    if device.type == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        desc.update(
            {
                "gpu_name": torch.cuda.get_device_name(idx),
                "gpu_count": torch.cuda.device_count(),
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "cuda_version_runtime": torch.version.cuda,
            }
        )
    return desc
