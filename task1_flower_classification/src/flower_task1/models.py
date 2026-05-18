from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34
from torchvision.models.resnet import BasicBlock, ResNet

SUPPORTED_MODELS = ("resnet18", "resnet34", "se_resnet18", "se_resnet34", "cbam_resnet18", "cbam_resnet34")


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.avg_pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        maxv = torch.amax(x, dim=(2, 3), keepdim=True)
        return self.sigmoid(self.mlp(avg) + self.mlp(maxv))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv = torch.amax(x, dim=1, keepdim=True)
        attn = torch.cat([avg, maxv], dim=1)
        return self.sigmoid(self.conv(attn))


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module: channel attention followed by spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x


class SEBasicBlock(BasicBlock):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: type[nn.Module] | None = None,
        reduction: int = 16,
    ) -> None:
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        self.se = SEBlock(planes * self.expansion, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class CBAMBasicBlock(BasicBlock):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: type[nn.Module] | None = None,
        reduction: int = 16,
    ) -> None:
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        self.cbam = CBAMBlock(planes * self.expansion, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def _base_resnet_factory(model_name: str, pretrained: bool) -> nn.Module:
    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return resnet18(weights=weights)
    if model_name == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        return resnet34(weights=weights)
    raise ValueError(f"Unsupported base model: {model_name}")


def _make_attention_resnet(model_name: str, pretrained: bool, num_classes: int) -> Tuple[nn.Module, Dict[str, object]]:
    if model_name.endswith("resnet18"):
        layers = [2, 2, 2, 2]
        base_name = "resnet18"
    elif model_name.endswith("resnet34"):
        layers = [3, 4, 6, 3]
        base_name = "resnet34"
    else:
        raise ValueError(f"Unsupported attention model: {model_name}")

    block = SEBasicBlock if model_name.startswith("se_") else CBAMBasicBlock
    # Build as 1000-class first if pretrained, so every shared tensor shape matches ImageNet ResNet.
    model = ResNet(block, layers, num_classes=1000 if pretrained else num_classes)
    load_report: Dict[str, object] = {"loaded_from_imagenet": False}
    if pretrained:
        source_model = _base_resnet_factory(base_name, pretrained=True)
        load_report = load_matching_state_dict(model, source_model.state_dict())
        load_report["loaded_from_imagenet"] = True
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model, load_report


def load_matching_state_dict(model: nn.Module, source_state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """Load only parameters whose keys and tensor shapes match.

    This is useful for SE/CBAM ResNet variants: normal ResNet parameters are loaded from
    ImageNet, while new attention modules stay randomly initialized.
    """
    own_state = model.state_dict()
    matched = {k: v for k, v in source_state.items() if k in own_state and own_state[k].shape == v.shape}
    skipped = sorted([k for k in source_state if k not in matched])
    missing, unexpected = model.load_state_dict(matched, strict=False)
    return {
        "loaded_tensors": len(matched),
        "skipped_source_tensors": skipped,
        "missing_model_tensors": list(missing),
        "unexpected_tensors": list(unexpected),
    }


def build_model(model_name: str, num_classes: int = 102, pretrained: bool = True) -> Tuple[nn.Module, Dict[str, object]]:
    model_name = model_name.lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Supported: {SUPPORTED_MODELS}")

    if model_name in ("resnet18", "resnet34"):
        model = _base_resnet_factory(model_name, pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        report = {"loaded_from_imagenet": bool(pretrained), "loaded_tensors": "torchvision_builtin" if pretrained else 0}
        return model, report

    return _make_attention_resnet(model_name, pretrained=pretrained, num_classes=num_classes)


def parameter_groups(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
) -> List[Dict[str, object]]:
    """Create parameter groups for fine-tuning.

    The classifier head is always newly initialized and therefore uses lr_head.
    In SE/CBAM variants, the attention modules are also newly initialized, so they
    share lr_head while the ImageNet-loaded convolutional backbone uses lr_backbone.
    """
    new_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_new_module = name.startswith("fc.") or ".se." in name or ".cbam." in name
        if is_new_module:
            new_params.append(param)
        else:
            backbone_params.append(param)
    groups: List[Dict[str, object]] = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay, "group_name": "backbone"})
    if new_params:
        groups.append({"params": new_params, "lr": lr_head, "weight_decay": weight_decay, "group_name": "head_or_attention"})
    return groups


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
