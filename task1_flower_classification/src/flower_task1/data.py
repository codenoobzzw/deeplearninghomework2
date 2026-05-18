from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Flowers102

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 102


def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train/eval transforms suitable for ImageNet-pretrained ResNet models."""
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.55, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def get_datasets(data_dir: str | Path, img_size: int = 224, download: bool = True) -> Dict[str, Dataset]:
    """Create official train/val/test splits for Oxford Flowers102."""
    data_dir = Path(data_dir)
    train_tf, eval_tf = build_transforms(img_size)
    return {
        "train": Flowers102(root=data_dir, split="train", transform=train_tf, download=download),
        "val": Flowers102(root=data_dir, split="val", transform=eval_tf, download=download),
        "test": Flowers102(root=data_dir, split="test", transform=eval_tf, download=download),
    }


def make_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    download: bool = True,
) -> Dict[str, DataLoader]:
    datasets = get_datasets(data_dir=data_dir, img_size=img_size, download=download)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
    }


def _labels_of_flowers102(ds: Dataset) -> list[int]:
    # torchvision Flowers102 stores labels as a private list. This helper avoids loading every image.
    labels = getattr(ds, "_labels", None)
    if labels is None:
        labels = [int(ds[i][1]) for i in range(len(ds))]
    return [int(x) for x in labels]


def dataset_summary(data_dir: str | Path, img_size: int = 224, download: bool = True) -> Dict[str, object]:
    """Return sizes and class-count information for the three official splits."""
    datasets = get_datasets(data_dir=data_dir, img_size=img_size, download=download)
    summary: Dict[str, object] = {"num_classes": NUM_CLASSES, "splits": {}}
    for split, ds in datasets.items():
        labels = _labels_of_flowers102(ds)
        counter = Counter(labels)
        summary["splits"][split] = {
            "num_images": len(ds),
            "num_classes_observed": len(counter),
            "min_images_per_class": min(counter.values()) if counter else 0,
            "max_images_per_class": max(counter.values()) if counter else 0,
        }
    return summary


def save_dataset_summary(path: str | Path, data_dir: str | Path, img_size: int = 224, download: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = dataset_summary(data_dir=data_dir, img_size=img_size, download=download)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
