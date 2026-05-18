from __future__ import annotations

import importlib
import platform
import sys


def check_import(name: str) -> None:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: OK ({version})")
    except Exception as exc:  # pragma: no cover
        print(f"{name}: FAILED -> {exc}")


def main() -> None:
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    for name in ["torch", "torchvision", "ultralytics", "cv2", "yaml", "pandas", "matplotlib", "numpy"]:
        check_import(name)

    try:
        import torch

        print("torch cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("cuda device count:", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                print(f"cuda:{i} -> {torch.cuda.get_device_name(i)}")
    except Exception as exc:
        print("CUDA check failed:", exc)


if __name__ == "__main__":
    main()
