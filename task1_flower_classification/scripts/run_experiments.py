#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def add_arg(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def bool_pretrained_arg(cmd: list[str], pretrained: bool) -> None:
    cmd.append("--pretrained" if pretrained else "--no-pretrained")


def build_command(defaults: Dict[str, Any], exp: Dict[str, Any], args: argparse.Namespace) -> list[str]:
    cfg = {**defaults, **exp}
    if args.epochs_override is not None:
        cfg["epochs"] = args.epochs_override
    if args.batch_size_override is not None:
        cfg["batch_size"] = args.batch_size_override
    if args.tracker is not None:
        cfg["tracker"] = args.tracker
    if args.wandb_mode is not None:
        cfg["wandb_mode"] = args.wandb_mode

    cmd = [sys.executable, str(TRAIN_SCRIPT)]
    add_arg(cmd, "--data-dir", args.data_dir)
    add_arg(cmd, "--output-dir", args.output_dir)
    add_arg(cmd, "--exp-name", cfg["name"])
    add_arg(cmd, "--model", cfg.get("model", "resnet18"))
    bool_pretrained_arg(cmd, bool(cfg.get("pretrained", True)))
    add_arg(cmd, "--epochs", cfg.get("epochs", 30))
    add_arg(cmd, "--batch-size", cfg.get("batch_size", 32))
    add_arg(cmd, "--num-workers", args.num_workers)
    add_arg(cmd, "--lr-head", cfg.get("lr_head", 1e-3))
    add_arg(cmd, "--lr-backbone", cfg.get("lr_backbone", 1e-4))
    add_arg(cmd, "--weight-decay", cfg.get("weight_decay", defaults.get("weight_decay", 1e-4)))
    add_arg(cmd, "--optimizer", cfg.get("optimizer", defaults.get("optimizer", "adamw")))
    add_arg(cmd, "--scheduler", cfg.get("scheduler", defaults.get("scheduler", "cosine")))
    add_arg(cmd, "--label-smoothing", cfg.get("label_smoothing", defaults.get("label_smoothing", 0.1)))
    add_arg(cmd, "--seed", cfg.get("seed", defaults.get("seed", 42)))
    add_arg(cmd, "--tracker", cfg.get("tracker", "none"))
    add_arg(cmd, "--wandb-project", cfg.get("wandb_project", defaults.get("wandb_project", "dl-hw2-task1-flowers102")))
    add_arg(cmd, "--wandb-mode", cfg.get("wandb_mode", defaults.get("wandb_mode", "offline")))
    add_arg(cmd, "--swanlab-project", cfg.get("swanlab_project", defaults.get("swanlab_project", "dl-hw2-task1-flowers102")))
    add_arg(cmd, "--patience", cfg.get("patience", defaults.get("patience", 0)))
    if args.device:
        add_arg(cmd, "--device", args.device)
    if args.no_download:
        cmd.append("--no-download")
    if args.amp or bool(cfg.get("amp", False)):
        cmd.append("--amp")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the planned ablation/hyperparameter experiments sequentially.")
    parser.add_argument("--plan", type=str, default="configs/experiment_plan.yaml")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--tracker", type=str, choices=("none", "wandb", "swanlab"), default=None)
    parser.add_argument("--wandb-mode", type=str, choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--epochs-override", type=int, default=None, help="Use this for a quick smoke test.")
    parser.add_argument("--batch-size-override", type=int, default=None, help="Use this if GPU memory is limited.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = REPO_ROOT / plan_path
    with plan_path.open("r", encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    defaults = plan.get("defaults", {})
    experiments = plan.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in plan.")

    for idx, exp in enumerate(experiments, start=1):
        name = exp["name"]
        metrics_path = Path(args.output_dir) / name / "metrics_best.json"
        if args.skip_existing and metrics_path.exists():
            print(f"[{idx}/{len(experiments)}] Skip existing experiment: {name}")
            continue
        cmd = build_command(defaults, exp, args)
        print("=" * 100)
        print(f"[{idx}/{len(experiments)}] {name}")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        if result.returncode != 0:
            print(f"[ERROR] Experiment {name} failed with return code {result.returncode}")
            if not args.continue_on_error:
                sys.exit(result.returncode)

    print("[DONE] Experiment runner finished. Run scripts/summarize_results.py next.")


if __name__ == "__main__":
    main()
