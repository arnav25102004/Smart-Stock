"""
Overlay plot: Actual vs AT-CNN (MSE) vs AT-CNN (RAL) for GROCERY I.

Visually demonstrates the RAL loss function's bias toward over-prediction
(safer for retail — avoids stockouts) relative to symmetric MSE training.

Usage:
    python src/forecasting/plot_ral_comparison.py
"""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

RESULTS_DIR = "results/tables"
PLOTS_DIR   = "results/plots"
FAMILY      = "GROCERY_I"

os.makedirs(PLOTS_DIR, exist_ok=True)


def load_preds(variant):
    path = os.path.join(RESULTS_DIR, f"forecast_atcnn_{variant}_{FAMILY}.csv")
    if not os.path.exists(path):
        print(f"[ERROR] Missing: {path}")
        sys.exit(1)
    return pd.read_csv(path)


def main():
    df_mse = load_preds("mse")
    df_ral = load_preds("ral")

    actual    = df_mse["actual"].values
    pred_mse  = df_mse["predicted"].values
    pred_ral  = df_ral["predicted"].values
    n         = len(actual)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(actual,   label="Actual Sales",      color="#1f77b4", linewidth=1.6)
    ax.plot(pred_mse, label="AT-CNN (MSE)",       color="#ff7f0e", linewidth=1.4,
            linestyle="--", alpha=0.9)
    ax.plot(pred_ral, label="AT-CNN (RAL)",       color="#2ca02c", linewidth=1.4,
            linestyle="--", alpha=0.9)

    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Sales (Units)")
    ax.set_title("Actual vs Predicted — AT-CNN MSE vs RAL (GROCERY I)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = os.path.join(PLOTS_DIR, "atcnn_ral_vs_mse_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")

    # ── Console statistics ────────────────────────────────────────────────────
    mean_actual  = np.mean(actual)
    mean_mse     = np.mean(pred_mse)
    mean_ral     = np.mean(pred_ral)
    bias_mse     = mean_mse  - mean_actual   # +ve = over-prediction
    bias_ral     = mean_ral  - mean_actual
    bias_diff    = mean_ral  - mean_mse      # how much more RAL over-predicts vs MSE

    under_mse = np.sum(pred_mse < actual)
    under_ral = np.sum(pred_ral < actual)
    over_mse  = np.sum(pred_mse > actual)
    over_ral  = np.sum(pred_ral > actual)

    def srr(y_true, y_pred, threshold=0.9):
        return 100 * np.sum(y_pred < y_true * threshold) / len(y_true)

    print()
    print("=" * 56)
    print("AT-CNN MSE vs RAL — GROCERY I  (n={:,} test days)".format(n))
    print("=" * 56)
    print(f"  Mean actual sales        : {mean_actual:>10.2f} units")
    print(f"  Mean prediction (MSE)    : {mean_mse:>10.2f} units")
    print(f"  Mean prediction (RAL)    : {mean_ral:>10.2f} units")
    print()
    print(f"  Bias vs actual (MSE)     : {bias_mse:>+10.2f} units  "
          f"({'over' if bias_mse >= 0 else 'under'}-predicts on average)")
    print(f"  Bias vs actual (RAL)     : {bias_ral:>+10.2f} units  "
          f"({'over' if bias_ral >= 0 else 'under'}-predicts on average)")
    print(f"  RAL excess over-pred vs MSE: {bias_diff:>+8.2f} units")
    print()
    print(f"  Days MSE under-predicted : {under_mse:>5} / {n}  "
          f"({100*under_mse/n:.1f}%)")
    print(f"  Days RAL under-predicted : {under_ral:>5} / {n}  "
          f"({100*under_ral/n:.1f}%)")
    print(f"  Days MSE over-predicted  : {over_mse:>5} / {n}  "
          f"({100*over_mse/n:.1f}%)")
    print(f"  Days RAL over-predicted  : {over_ral:>5} / {n}  "
          f"({100*over_ral/n:.1f}%)")
    print()
    print(f"  Stockout Risk Rate (MSE) : {srr(actual, pred_mse):.2f}%")
    print(f"  Stockout Risk Rate (RAL) : {srr(actual, pred_ral):.2f}%")
    print("=" * 56)


if __name__ == "__main__":
    main()
