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
        raise ValueError(f"Invalid plan file: {path}")
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
    parser = argparse.ArgumentParser(description="Run YOLOv8 experiment plan.")
    parser.add_argument("--plan", type=Path, default=Path("configs/task2_yolo_plan.yaml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--batch-override", type=int, default=None)
    parser.add_argument("--imgsz-override", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--tracker", choices=["none", "wandb"], default="none")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    args = parser.parse_args()

    args.data = args.data.resolve()
    args.project = args.project.resolve()
    args.project.mkdir(parents=True, exist_ok=True)
    logs_dir = args.project.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    experiments = load_plan(args.plan)
    script = Path(__file__).resolve().parent / "train_yolo.py"
    failures: list[tuple[str, int]] = []

    for exp in experiments:
        name = str(exp["name"])
        best = args.project / name / "weights" / "best.pt"
        if args.skip_existing and best.exists():
            print(f"[SKIP] {name}: {best} exists")
            continue

        cmd = [
            sys.executable,
            str(script),
            "--data",
            str(args.data),
            "--model",
            str(exp.get("model", "yolov8n.pt")),
            "--epochs",
            str(args.epochs_override or exp.get("epochs", 80)),
            "--imgsz",
            str(args.imgsz_override or exp.get("imgsz", 640)),
            "--batch",
            str(args.batch_override or exp.get("batch", 16)),
            "--device",
            str(args.device),
            "--workers",
            str(args.workers),
            "--project",
            str(args.project),
            "--name",
            name,
            "--lr0",
            str(exp.get("lr0", 0.01)),
            "--lrf",
            str(exp.get("lrf", 0.01)),
            "--optimizer",
            str(exp.get("optimizer", "SGD")),
            "--patience",
            str(exp.get("patience", 20)),
            "--tracker",
            args.tracker,
            "--wandb-mode",
            args.wandb_mode,
        ]
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
    print("All requested experiments finished.")


if __name__ == "__main__":
    main()
