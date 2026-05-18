from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def summarize_run(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics_best.json"
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.csv"
    if not metrics_path.exists():
        return None
    metrics = load_json(metrics_path)
    config = load_json(config_path)
    row = {
        "experiment": run_dir.name,
        "loss": config.get("loss"),
        "epochs_config": config.get("epochs"),
        "batch_size": config.get("batch_size"),
        "image_size": config.get("image_size"),
        "base_channels": config.get("base_channels"),
        "lr": config.get("lr"),
        "weight_decay": config.get("weight_decay"),
        "best_epoch": metrics.get("epoch"),
        "best_val_miou": metrics.get("val_miou"),
        "best_val_pixel_acc": metrics.get("val_pixel_acc"),
        "best_val_mean_acc": metrics.get("val_mean_acc"),
        "best_val_loss": metrics.get("val_loss"),
        "best_train_loss": metrics.get("train_loss"),
        "best_pt": str(run_dir / "best.pt"),
        "last_pt": str(run_dir / "last.pt"),
    }
    if history_path.exists():
        try:
            row["epochs_logged"] = len(pd.read_csv(history_path))
        except Exception:
            pass
    return row


def plot_curves(runs_dir: Path, out_dir: Path) -> None:
    specs = [
        ("train_loss", "train_loss_curves.png", "Train loss"),
        ("val_loss", "val_loss_curves.png", "Validation loss"),
        ("val_miou", "val_miou_curves.png", "Validation mIoU"),
        ("val_pixel_acc", "val_pixel_acc_curves.png", "Validation pixel accuracy"),
    ]
    for metric, filename, title in specs:
        plt.figure(figsize=(10, 6))
        plotted = False
        for history_path in sorted(runs_dir.glob("*/history.csv")):
            df = pd.read_csv(history_path)
            if metric not in df.columns:
                continue
            plt.plot(df["epoch"], df[metric], label=history_path.parent.name)
            plotted = True
        if plotted:
            plt.title(title)
            plt.xlabel("epoch")
            plt.ylabel(metric)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / filename, dpi=200)
        plt.close()


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Task 3 U-Net Loss Comparison Summary", ""]
    if df.empty:
        lines.append("No completed runs found.")
    else:
        sorted_df = df.sort_values("best_val_miou", ascending=False, na_position="last") if "best_val_miou" in df.columns else df
        lines.append(sorted_df.to_markdown(index=False))
        lines.append("")
        best = sorted_df.iloc[0].to_dict()
        lines.append("## Best run")
        lines.append("")
        for k in ["experiment", "loss", "best_epoch", "best_val_miou", "best_val_pixel_acc", "best_val_loss", "best_pt"]:
            lines.append(f"- {k}: {best.get(k)}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize U-Net loss experiments.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
        row = summarize_run(run_dir)
        if row:
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "best_val_miou" in df.columns:
        df = df.sort_values("best_val_miou", ascending=False, na_position="last")
    df.to_csv(args.out_dir / "summary.csv", index=False)
    write_markdown(df, args.out_dir / "summary.md")

    if not df.empty and "best_val_miou" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.bar(df["experiment"], df["best_val_miou"])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("best val mIoU")
        plt.title("U-Net loss comparison")
        plt.tight_layout()
        plt.savefig(args.out_dir / "summary_bar.png", dpi=200)
        plt.close()
    plot_curves(args.runs_dir, args.out_dir)
    print(f"Wrote summary to {args.out_dir}")


if __name__ == "__main__":
    main()
