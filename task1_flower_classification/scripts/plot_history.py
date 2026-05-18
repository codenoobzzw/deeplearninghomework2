#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one run's train/val loss and validation accuracy curves.")
    parser.add_argument("--history", type=str, required=True, help="Path to history.csv")
    parser.add_argument("--out-dir", type=str, default="results/one_run_plots")
    args = parser.parse_args()

    hist = pd.read_csv(args.history)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, filename in [
        ("train_loss", "Train loss", "train_loss.png"),
        ("val_loss", "Validation loss", "val_loss.png"),
        ("val_acc1", "Validation Accuracy@1 (%)", "val_acc1.png"),
    ]:
        if metric not in hist:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(hist["epoch"], hist[metric], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()

    print(f"[DONE] Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
