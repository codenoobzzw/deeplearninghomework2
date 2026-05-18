from __future__ import annotations

import numpy as np
import torch


class SegmentationMeter:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
        gt = target.detach().cpu().numpy().astype(np.int64)
        mask = gt != self.ignore_index
        gt = gt[mask]
        pred = pred[mask]
        valid = (gt >= 0) & (gt < self.num_classes) & (pred >= 0) & (pred < self.num_classes)
        gt = gt[valid]
        pred = pred[valid]
        if gt.size == 0:
            return
        inds = self.num_classes * gt + pred
        cm = np.bincount(inds, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion += cm

    def compute(self) -> dict[str, float | list[float]]:
        cm = self.confusion.astype(np.float64)
        tp = np.diag(cm)
        pos_gt = cm.sum(axis=1)
        pos_pred = cm.sum(axis=0)
        union = pos_gt + pos_pred - tp
        iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
        miou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0
        pixel_acc = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0
        mean_acc_per_class = np.divide(tp, pos_gt, out=np.full_like(tp, np.nan), where=pos_gt > 0)
        mean_acc = float(np.nanmean(mean_acc_per_class)) if np.any(~np.isnan(mean_acc_per_class)) else 0.0
        return {
            "miou": miou,
            "pixel_acc": pixel_acc,
            "mean_acc": mean_acc,
            "class_iou": [float(x) if not np.isnan(x) else None for x in iou],
        }
