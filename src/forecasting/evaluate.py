"""
Master evaluation script for the forecasting experiment.

Loads all per-model summary CSVs, compiles the master comparison table,
and generates all publication-ready plots.

Usage:
    python src/forecasting/evaluate.py
    (Run after all model_*.py scripts have completed.)
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import visualization as viz

RESULTS_DIR = "results/tables"
PLOTS_DIR = "results/plots"
PROCESSED_DIR = "data/processed"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_summary(filename):
    """Load a model summary CSV, returning None if the file does not exist.

    Args:
        filename: Filename (without directory prefix) of the summary CSV.

    Returns:
        Single-row DataFrame or None.
    """
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"  [WARN] Missing summary: {path}")
    return None


def build_master_table():
    """Combine per-model summaries into the main forecasting comparison table.

    Includes Loss and SRR (%) columns sourced from full_comparison_with_ral.csv
    when available. All 6 models are included: Moving Average, Prophet, LSTM,
    CNN, AT-CNN (MSE), AT-CNN (RAL).

    Returns:
        DataFrame with one row per model.
    """
    summaries = [
        load_summary("forecast_moving_avg_summary.csv"),
        load_summary("forecast_prophet_summary.csv"),
        load_summary("forecast_lstm_summary.csv"),
        load_summary("forecast_cnn_summary.csv"),
        load_summary("forecast_atcnn_mse_summary.csv"),
        load_summary("forecast_atcnn_ral_summary.csv"),
    ]
    frames = [s for s in summaries if s is not None]
    if not frames:
        print("[ERROR] No model summaries found. Run all model scripts first.")
        sys.exit(1)
    master = pd.concat(frames, ignore_index=True)

    # Add Loss column
    loss_map = {
        "Moving Average (7-day)": "MSE",
        "Moving Average":         "MSE",
        "Prophet":                "MSE",
        "LSTM":                   "MSE",
        "CNN":                    "MSE",
        "AT-CNN (MSE)":           "MSE",
        "AT-CNN (RAL)":           "RAL",
    }
    master.insert(1, "Loss", master["Model"].map(loss_map).fillna("MSE"))

    # Merge SRR from full_comparison_with_ral.csv when available
    ral_path = os.path.join(RESULTS_DIR, "full_comparison_with_ral.csv")
    if os.path.exists(ral_path):
        ral_df = pd.read_csv(ral_path)[["Model", "SRR (%)"]].copy()
        # Normalise Moving Average name variant
        ral_df["Model"] = ral_df["Model"].replace(
            {"Moving Average": "Moving Average (7-day)"}
        )
        master = master.merge(ral_df, on="Model", how="left")

    # Reorder columns for the paper table
    ordered = ["Model", "Loss", "MAPE (%)", "RMSE", "MAE",
               "SRR (%)", "Training_Time_s", "Inference_Time_ms", "Model_Size_KB"]
    master = master[[c for c in ordered if c in master.columns]]
    return master


def plot_actual_vs_predicted_all(families):
    """Generate actual-vs-predicted line charts for each model × family pair.

    Args:
        families: List of product family names.
    """
    models = {
        "Moving_Average":   "Moving Average",
        "Prophet":          "Prophet",
        "lstm":             "LSTM",
        "cnn":              "CNN",
        "atcnn_mse":        "AT-CNN (MSE)",
        "atcnn_ral":        "AT-CNN (RAL)",
    }
    for fam in families:
        safe = fam.replace(' ', '_')
        for key, label in models.items():
            csv_map = {
                "Moving_Average": f"forecast_moving_avg_{safe}.csv",
                "Prophet":        f"forecast_prophet_{safe}.csv",
                "lstm":           f"forecast_lstm_{safe}.csv",
                "cnn":            f"forecast_cnn_{safe}.csv",
                "atcnn_mse":      f"forecast_atcnn_mse_{safe}.csv",
                "atcnn_ral":      f"forecast_atcnn_ral_{safe}.csv",
            }
            path = os.path.join(RESULTS_DIR, csv_map[key])
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            viz.plot_actual_vs_predicted(
                df["actual"].values,
                df["predicted"].values,
                label,
                fam,
                f"actual_vs_pred_{key}_{safe}"
            )
    print("  Saved actual-vs-predicted plots.")


def plot_training_losses(families):
    """Generate training/validation loss curves for LSTM and CNN.

    Args:
        families: List of product family names.
    """
    for arch, label in [("lstm", "LSTM"), ("cnn", "CNN"),
                        ("atcnn_mse", "AT-CNN (MSE)"), ("atcnn_ral", "AT-CNN (RAL)")]:
        for fam in families:
            safe = fam.replace(' ', '_')
            path = os.path.join(RESULTS_DIR, f"{arch}_history_{safe}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            viz.plot_training_loss(
                df["train_loss"].values,
                df["val_loss"].values,
                label,
                f"loss_{arch}_{safe}"
            )
    print("  Saved training loss curves.")


def main():
    print("=" * 60)
    print("STEP 7: Master Forecasting Evaluation")
    print("=" * 60)

    meta_path = os.path.join(PROCESSED_DIR, "split_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            families = json.load(f)["target_columns"]
    else:
        families = ["GROCERY I", "DAIRY", "BEVERAGES", "CLEANING"]

    print("\n  [1/5] Building master comparison table...")
    master = build_master_table()
    master_path = os.path.join(RESULTS_DIR, "forecasting_comparison.csv")
    master.to_csv(master_path, index=False)
    print(f"\n{master.to_string(index=False)}")

    model_names = master["Model"].tolist()
    mape_vals = master["MAPE (%)"].tolist()
    rmse_vals = master["RMSE"].tolist()
    size_vals = master.get("Model_Size_KB", pd.Series([0] * len(master))).tolist()

    print("\n  [2/5] Plotting actual vs predicted...")
    plot_actual_vs_predicted_all(families)

    print("\n  [3/5] Plotting metric comparison bars...")
    viz.plot_metric_comparison_bar(model_names, mape_vals, "MAPE (%)", "forecast_mape_comparison")
    viz.plot_metric_comparison_bar(model_names, rmse_vals, "RMSE", "forecast_rmse_comparison")
    print("  Saved metric comparison bar charts.")

    print("\n  [4/5] Plotting model size vs accuracy...")
    # Filter out models with zero size (moving average has no stored model)
    valid = [(n, s, m) for n, s, m in zip(model_names, size_vals, mape_vals) if s > 0]
    if valid:
        ns, ss, ms = zip(*valid)
        viz.plot_model_size_vs_accuracy(list(ns), list(ss), list(ms), "forecast_size_vs_accuracy")
        print("  Saved size vs accuracy scatter plot.")

    print("\n  [5/5] Plotting compression comparison...")
    comp_path = os.path.join(RESULTS_DIR, "compression_comparison.csv")
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        for arch in ["LSTM", "CNN"]:
            sub = comp_df[comp_df["Architecture"] == arch]
            if sub.empty:
                continue
            avg_by_version = sub.groupby("Version")["Size_KB"].mean()
            full_sizes = [avg_by_version.get("Full (.h5)", 0)] * 2
            compressed_dynamic = avg_by_version.get("TFLite Dynamic Quant", 0)
            compressed_f16 = avg_by_version.get("TFLite Float16 Quant", 0)
            viz.plot_compression_comparison(
                [f"{arch} Dynamic", f"{arch} Float16"],
                [avg_by_version.get("Full (.h5)", 0)] * 2,
                [compressed_dynamic, compressed_f16],
                f"compression_{arch.lower()}"
            )
        print("  Saved compression comparison charts.")
    else:
        print("  [SKIP] compression_comparison.csv not found.")

    plot_training_losses(families)

    print(f"\n  Saved: {master_path}")
    print(f"  All plots saved to: {PLOTS_DIR}/")
    print("  [DONE] Forecasting evaluation complete.")


if __name__ == "__main__":
    main()
