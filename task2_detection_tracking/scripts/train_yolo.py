from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from ultralytics import YOLO


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def has_test_split(data_yaml: Path) -> bool:
    cfg = load_yaml(data_yaml)
    return "test" in cfg and cfg.get("test") not in (None, "")


def to_serializable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)


def save_metrics(metrics: Any, out_path: Path) -> None:
    result: dict[str, Any] = {}
    if hasattr(metrics, "results_dict"):
        result.update(getattr(metrics, "results_dict"))
    elif isinstance(metrics, dict):
        result.update(metrics)
    result = {str(k): to_serializable(v) for k, v in result.items()}
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def maybe_log_wandb(run_dir: Path, project: str, mode: str) -> None:
    try:
        import wandb
    except Exception as exc:
        print(f"wandb import failed, skip logging: {exc}")
        return
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"No results.csv at {csv_path}, skip wandb logging.")
        return
    os.environ["WANDB_MODE"] = mode
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    run = wandb.init(project=project, name=run_dir.name, dir=str(run_dir), mode=mode)
    for _, row in df.iterrows():
        record = {str(k).strip().replace("/", "_"): float(v) for k, v in row.items() if pd.notna(v)}
        step = int(record.get("epoch", len(record)))
        wandb.log(record, step=step)
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Road Vehicle Images Dataset.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model, e.g. yolov8n.pt or yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="yolov8n_baseline")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument(
        "--optimizer",
        default="SGD",
        help="Optimizer passed to Ultralytics. Use an explicit optimizer so lr0/lrf are not ignored by optimizer=auto.",
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--tracker", choices=["none", "wandb"], default="none")
    parser.add_argument("--wandb-project", default="dl-hw2-task2-yolo")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    args = parser.parse_args()

    args.data = args.data.resolve()
    args.project = args.project.resolve()
    args.project.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    train_result = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        patience=args.patience,
        seed=args.seed,
        cache=args.cache,
        amp=args.amp,
        plots=True,
        exist_ok=True,
    )
    run_dir = Path(getattr(train_result, "save_dir", args.project / args.name))
    if not run_dir.exists():
        run_dir = args.project / args.name

    best = run_dir / "weights" / "best.pt"
    if best.exists():
        best_model = YOLO(str(best))
        split = "test" if has_test_split(args.data) else "val"
        try:
            metrics = best_model.val(
                data=str(args.data),
                split=split,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                plots=True,
                project=str(run_dir),
                name=f"{split}_eval",
                exist_ok=True,
            )
            save_metrics(metrics, run_dir / f"{split}_metrics.json")
        except Exception as exc:
            print(f"Validation on split={split} failed: {exc}")
    else:
        print(f"Best checkpoint not found: {best}")

    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    (run_dir / "train_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.tracker == "wandb" and args.wandb_mode != "disabled":
        maybe_log_wandb(run_dir, args.wandb_project, args.wandb_mode)

    print(f"Run directory: {run_dir}")
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
