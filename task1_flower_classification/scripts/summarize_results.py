#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows(runs_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for metrics_path in sorted(runs_dir.glob("*/metrics_best.json")):
        metrics = load_json(metrics_path)
        row = {
            "exp_name": metrics.get("exp_name", metrics_path.parent.name),
            "model": metrics.get("model", ""),
            "pretrained": metrics.get("pretrained", ""),
            "best_epoch": metrics.get("best_epoch", ""),
            "best_val_acc1": metrics.get("best_val_acc1", ""),
            "test_acc1": metrics.get("test_acc1", ""),
            "test_acc5": metrics.get("test_acc5", ""),
            "lr_head": metrics.get("lr_head", ""),
            "lr_backbone": metrics.get("lr_backbone", ""),
            "batch_size": metrics.get("batch_size", ""),
            "optimizer": metrics.get("optimizer", ""),
            "scheduler": metrics.get("scheduler", ""),
            "best_checkpoint": metrics.get("best_checkpoint", str(metrics_path.parent / "best.pt")),
        }
        rows.append(row)
    return rows


def plot_summary(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty or "best_val_acc1" not in df:
        return
    plot_df = df.copy()
    plot_df["best_val_acc1"] = pd.to_numeric(plot_df["best_val_acc1"], errors="coerce")
    plot_df = plot_df.dropna(subset=["best_val_acc1"]).sort_values("best_val_acc1")
    if plot_df.empty:
        return
    labels = plot_df["exp_name"].astype(str).tolist()
    values = plot_df["best_val_acc1"].tolist()
    height = max(4, 0.35 * len(plot_df) + 2)
    plt.figure(figsize=(10, height))
    plt.barh(labels, values)
    plt.xlabel("Best validation Accuracy@1 (%)")
    plt.ylabel("Experiment")
    plt.title("Flowers102 experiment comparison")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_histories(runs_dir: Path, out_dir: Path) -> None:
    histories = []
    for hist_path in sorted(runs_dir.glob("*/history.csv")):
        try:
            df = pd.read_csv(hist_path)
            if not df.empty:
                df["exp_name"] = hist_path.parent.name
                histories.append(df)
        except Exception:
            continue
    if not histories:
        return
    all_hist = pd.concat(histories, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, filename in [
        ("val_acc1", "Validation Accuracy@1 (%)", "val_accuracy_curves.png"),
        ("train_loss", "Training loss", "train_loss_curves.png"),
        ("val_loss", "Validation loss", "val_loss_curves.png"),
    ]:
        if metric not in all_hist.columns:
            continue
        plt.figure(figsize=(10, 6))
        for exp_name, group in all_hist.groupby("exp_name"):
            plt.plot(group["epoch"], group[metric], label=str(exp_name))
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("No finished experiments found.\n", encoding="utf-8")
        return
    display_df = df.copy()
    for col in ["best_val_acc1", "test_acc1", "test_acc5"]:
        if col in display_df:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(2)
    content = ["# Experiment Summary", "", display_df.to_markdown(index=False), ""]
    best_col = "test_acc1" if display_df["test_acc1"].notna().any() else "best_val_acc1"
    numeric = pd.to_numeric(display_df[best_col], errors="coerce")
    if numeric.notna().any():
        best_idx = numeric.idxmax()
        content.extend(
            [
                "## Best run",
                "",
                f"Best by `{best_col}`: `{display_df.loc[best_idx, 'exp_name']}` "
                f"({best_col}={display_df.loc[best_idx, best_col]}).",
                "",
            ]
        )
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect all run metrics into CSV/Markdown and plots.")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    rows = collect_rows(runs_dir)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["test_acc1", "best_val_acc1"], ascending=False, na_position="last")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    write_markdown(df, out_dir / "summary.md")
    plot_summary(df, out_dir / "summary_bar.png")
    plot_histories(runs_dir, out_dir)

    print(f"[DONE] Found {len(df)} runs.")
    print(f"- CSV: {out_dir / 'summary.csv'}")
    print(f"- Markdown: {out_dir / 'summary.md'}")
    print(f"- Plots: {out_dir}")


if __name__ == "__main__":
    main()
