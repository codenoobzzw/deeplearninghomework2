from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    columns = list(df.columns)
    normalized = {c.lower().replace(" ", ""): c for c in columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "")
        if key in normalized:
            return normalized[key]
    for cand in candidates:
        key = cand.lower().replace(" ", "")
        for norm, original in normalized.items():
            if key in norm:
                return original
    return None


def load_metric_json(run_dir: Path) -> dict[str, Any]:
    for name in ["test_metrics.json", "val_metrics.json"]:
        p = run_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def summarize_run(run_dir: Path) -> dict[str, Any] | None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    df = clean_columns(pd.read_csv(csv_path))
    if df.empty:
        return None
    map5095_col = find_col(df, ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"])
    map50_col = find_col(df, ["metrics/mAP50(B)", "metrics/mAP50", "mAP50"])
    precision_col = find_col(df, ["metrics/precision(B)", "precision"])
    recall_col = find_col(df, ["metrics/recall(B)", "recall"])
    fitness_col = find_col(df, ["fitness"])
    target_col = map5095_col or map50_col or fitness_col
    best_idx = int(df[target_col].idxmax()) if target_col else len(df) - 1
    best_row = df.loc[best_idx]
    metrics_json = load_metric_json(run_dir)

    def val(col: str | None) -> float | None:
        if col is None:
            return None
        try:
            return float(best_row[col])
        except Exception:
            return None

    row = {
        "experiment": run_dir.name,
        "epochs_logged": int(len(df)),
        "best_epoch": int(best_row.get("epoch", best_idx)),
        "precision_B": val(precision_col),
        "recall_B": val(recall_col),
        "mAP50_B": val(map50_col),
        "mAP50_95_B": val(map5095_col),
        "fitness": val(fitness_col),
        "best_pt": str(run_dir / "weights" / "best.pt"),
        "last_pt": str(run_dir / "weights" / "last.pt"),
    }
    for k, v in metrics_json.items():
        safe_k = str(k).replace("/", "_").replace("(", "").replace(")", "")
        if isinstance(v, (int, float)):
            row[f"eval_{safe_k}"] = float(v)
    return row


def plot_curves(runs_dir: Path, out_dir: Path) -> None:
    curve_specs = [
        ("metrics/mAP50(B)", "mAP50_curves.png", "Validation mAP@0.50"),
        ("metrics/mAP50-95(B)", "map5095_curves.png", "Validation mAP@0.50:0.95"),
        ("train/box_loss", "box_loss_curves.png", "Train box loss"),
        ("val/box_loss", "val_box_loss_curves.png", "Validation box loss"),
    ]
    for target_name, filename, title in curve_specs:
        plt.figure(figsize=(10, 6))
        plotted = False
        for csv_path in sorted(runs_dir.glob("*/results.csv")):
            df = clean_columns(pd.read_csv(csv_path))
            col = find_col(df, [target_name, target_name.replace("(B)", "")])
            if col is None:
                continue
            x_col = find_col(df, ["epoch"])
            x = df[x_col] if x_col else range(len(df))
            plt.plot(x, df[col], label=csv_path.parent.name)
            plotted = True
        if plotted:
            plt.title(title)
            plt.xlabel("epoch")
            plt.ylabel(target_name)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / filename, dpi=200)
        plt.close()

    plt.figure(figsize=(10, 6))
    plotted = False
    for csv_path in sorted(runs_dir.glob("*/results.csv")):
        df = clean_columns(pd.read_csv(csv_path))
        x_col = find_col(df, ["epoch"])
        x = df[x_col] if x_col else range(len(df))
        train_col = find_col(df, ["train/box_loss"])
        val_col = find_col(df, ["val/box_loss"])
        if train_col:
            plt.plot(x, df[train_col], label=f"{csv_path.parent.name} train")
            plotted = True
        if val_col:
            plt.plot(x, df[val_col], linestyle="--", label=f"{csv_path.parent.name} val")
            plotted = True
    if plotted:
        plt.title("YOLO box loss curves")
        plt.xlabel("epoch")
        plt.ylabel("box loss")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curves.png", dpi=200)
    plt.close()


def write_markdown(summary: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Task 2 YOLOv8 Experiment Summary", ""]
    if summary.empty:
        lines.append("No runs found.")
    else:
        sort_col = "mAP50_95_B" if "mAP50_95_B" in summary.columns else "mAP50_B"
        summary_sorted = summary.sort_values(sort_col, ascending=False, na_position="last") if sort_col in summary else summary
        lines.append(summary_sorted.to_markdown(index=False))
        lines.append("")
        best = summary_sorted.iloc[0].to_dict()
        lines.append("## Best run")
        lines.append("")
        for k in ["experiment", "best_epoch", "precision_B", "recall_B", "mAP50_B", "mAP50_95_B", "best_pt"]:
            if k in best:
                lines.append(f"- {k}: {best[k]}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize YOLOv8 runs.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/detect"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
        row = summarize_run(run_dir)
        if row:
            rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        sort_col = "mAP50_95_B" if "mAP50_95_B" in summary.columns else "mAP50_B"
        if sort_col in summary.columns:
            summary = summary.sort_values(sort_col, ascending=False, na_position="last")
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    write_markdown(summary, args.out_dir / "summary.md")

    if not summary.empty and "mAP50_95_B" in summary.columns:
        plt.figure(figsize=(10, 6))
        plt.bar(summary["experiment"], summary["mAP50_95_B"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("mAP50-95")
        plt.title("YOLOv8 experiment comparison")
        plt.tight_layout()
        plt.savefig(args.out_dir / "summary_bar.png", dpi=200)
        plt.close()

    plot_curves(args.runs_dir, args.out_dir)
    print(f"Wrote summary to {args.out_dir}")


if __name__ == "__main__":
    main()
