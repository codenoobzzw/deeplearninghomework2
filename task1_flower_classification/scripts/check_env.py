#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from pathlib import Path


def version_of(pkg: str) -> str:
    try:
        mod = importlib.import_module(pkg)
        return getattr(mod, "__version__", "installed, version unknown")
    except Exception as exc:  # noqa: BLE001
        return f"not importable: {exc}"


def main() -> None:
    print("=== Python ===")
    print(sys.version)
    print("platform:", platform.platform())
    print("cwd:", Path.cwd())

    print("\n=== Packages ===")
    for pkg in ["torch", "torchvision", "scipy", "numpy", "pandas", "matplotlib", "wandb", "swanlab"]:
        print(f"{pkg}: {version_of(pkg)}")

    print("\n=== Torch / CUDA ===")
    try:
        import torch

        print("torch cuda available:", torch.cuda.is_available())
        print("torch cuda runtime:", torch.version.cuda)
        print("cuda device count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}, {props.total_memory / 1024**3:.2f} GB")
    except Exception as exc:  # noqa: BLE001
        print("torch cuda check failed:", exc)

    print("\n=== nvidia-smi ===")
    try:
        result = subprocess.run(["nvidia-smi"], check=False, text=True, capture_output=True, timeout=10)
        print(result.stdout if result.stdout else result.stderr)
    except Exception as exc:  # noqa: BLE001
        print("nvidia-smi not available:", exc)


if __name__ == "__main__":
    main()
