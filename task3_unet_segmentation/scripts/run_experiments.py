from __future__ import annotations

import argparse
from collections import deque
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_plan(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "experiments" not in data:
        raise ValueError(f"Invalid plan: {path}")
    return list(data["experiments"])


def print_log_tail(log_path: Path, lines: int = 80) -> None:
    if not log_path.exists():
        return
    tail: deque[str] = deque(maxlen=lines)
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            tail.append(line.rstrip("\n"))
    print(f"Last {len(tail)} log lines from {log_path}:")
    for line in tail:
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run U-Net loss comparison experiments.")
    parser.add_argument("--plan", type=Path, default=Path("configs/task3_unet_plan.yaml"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--batch-size-override", type=int, default=None)
    parser.add_argument("--image-size-override", type=int, default=None)
    parser.add_argument("--base-channels-override", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tracker", choices=["none", "wandb", "swanlab"], default="none")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()
    args.runs_dir = args.runs_dir.resolve()
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.runs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent / "train.py"
    failures: list[tuple[str, int]] = []
    for exp in load_plan(args.plan):
        name = str(exp["name"])
        out_dir = args.runs_dir / name
        if args.skip_existing and (out_dir / "best.pt").exists():
            print(f"[SKIP] {name}: best.pt exists")
            continue
        cmd = [
            sys.executable,
            str(script),
            "--data-dir",
            str(args.data_dir),
            "--loss",
            str(exp.get("loss", "ce")),
            "--epochs",
            str(args.epochs_override or exp.get("epochs", 80)),
            "--batch-size",
            str(args.batch_size_override or exp.get("batch_size", 8)),
            "--image-size",
            str(args.image_size_override or exp.get("image_size", 256)),
            "--base-channels",
            str(args.base_channels_override or exp.get("base_channels", 32)),
            "--lr",
            str(exp.get("lr", 3e-4)),
            "--weight-decay",
            str(exp.get("weight_decay", 1e-4)),
            "--ce-weight",
            str(exp.get("ce_weight", 1.0)),
            "--dice-weight",
            str(exp.get("dice_weight", 1.0)),
            "--num-workers",
            str(args.num_workers),
            "--output-dir",
            str(out_dir),
            "--device",
            args.device,
            "--tracker",
            args.tracker,
            "--wandb-mode",
            args.wandb_mode,
        ]
        if args.amp:
            cmd.append("--amp")
        print("\n" + "=" * 100)
        print("Running:", " ".join(cmd))
        log_path = logs_dir / f"{name}.log"
        print("Log:", log_path)
        print("=" * 100)
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            failures.append((name, proc.returncode))
            print(f"[FAILED] {name}: return code {proc.returncode}")
            print_log_tail(log_path)
            if not args.continue_on_error:
                raise SystemExit(proc.returncode)
    if failures:
        print("Failures:")
        for name, code in failures:
            print(f"  - {name}: {code}")
        raise SystemExit(1 if not args.continue_on_error else 0)
    print("All requested U-Net experiments finished.")


if __name__ == "__main__":
    main()
