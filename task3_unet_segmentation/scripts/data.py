from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import torch.nn.functional as F

CLASS_NAMES = ["sky", "tree", "road", "grass", "water", "building", "mountain", "foreground"]
IGNORE_INDEX = 255


def read_split_file(data_dir: Path, split: str) -> list[tuple[Path, Path]]:
    split_path = data_dir / "splits" / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    pairs: list[tuple[Path, Path]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        img, mask = line.split("\t")[:2]
        pairs.append((Path(img), Path(mask)))
    return pairs


class StanfordBackgroundDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = 256,
        augment: bool = False,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = int(image_size)
        self.augment = augment
        self.pairs = read_split_file(self.data_dir, split)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if np.random.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.rand() < 0.3:
            factor = float(np.random.uniform(0.8, 1.2))
            image = ImageEnhance.Brightness(image).enhance(factor)
        if np.random.rand() < 0.3:
            factor = float(np.random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        img_path, mask_path = self.pairs[index]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.augment:
            image, mask = self._augment(image, mask)
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)
        image_tensor = (image_tensor - self.mean) / self.std

        mask_arr = np.asarray(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_arr).long()
        return {"image": image_tensor, "mask": mask_tensor, "image_path": str(img_path), "mask_path": str(mask_path)}


def denormalize(image: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=image.device).view(3, 1, 1)
    std_t = torch.tensor(std, device=image.device).view(3, 1, 1)
    return torch.clamp(image * std_t + mean_t, 0.0, 1.0)
