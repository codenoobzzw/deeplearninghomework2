from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Manual multi-class Dice Loss for semantic segmentation.

    This implementation does not rely on any external segmentation loss library.
    It applies softmax to logits, converts target masks to one-hot tensors, ignores
    pixels with ignore_index, and averages Dice over classes.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError("logits must have shape [B, C, H, W]")
        if target.ndim != 3:
            raise ValueError("target must have shape [B, H, W]")

        probs = torch.softmax(logits, dim=1)
        valid = target != self.ignore_index
        safe_target = target.clone()
        safe_target[~valid] = 0
        one_hot = F.one_hot(safe_target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1).float()
        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        cardinality = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int, ce_weight: float = 1.0, dice_weight: float = 1.0, ignore_index: int = 255) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def build_loss(name: str, num_classes: int, ignore_index: int = 255, ce_weight: float = 1.0, dice_weight: float = 1.0) -> nn.Module:
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    if name == "dice":
        return DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
    if name in {"ce_dice", "combined", "ce+dice"}:
        return CombinedLoss(num_classes=num_classes, ce_weight=ce_weight, dice_weight=dice_weight, ignore_index=ignore_index)
    raise ValueError(f"Unknown loss: {name}")
