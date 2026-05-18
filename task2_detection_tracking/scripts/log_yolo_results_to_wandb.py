from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Log Ultralytics results.csv to wandb.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project", default="dl-hw2-task2-yolo")
    parser.add_argument("--mode", choices=["online", "offline", "disabled"], default="offline")
    args = parser.parse_args()

    csv_path = args.run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    os.environ["WANDB_MODE"] = args.mode
    import wandb

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    run = wandb.init(project=args.project, name=args.run_dir.name, dir=str(args.run_dir), mode=args.mode)
    for i, row in df.iterrows():
        record = {}
        for k, v in row.items():
            try:
                record[str(k).strip().replace("/", "_")] = float(v)
            except Exception:
                pass
        step = int(record.get("epoch", i))
        wandb.log(record, step=step)
    run.finish()
    print(f"Logged {len(df)} rows from {csv_path} to wandb mode={args.mode}")


if __name__ == "__main__":
    main()
