"""
Simple Moving Average (7-day) baseline for demand forecasting.

Predicts the next day's sales as the mean of the previous 7 actual sales days.
This is the simplest possible baseline and sets the lower bound for model quality.

Usage:
    python src/forecasting/model_moving_avg.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.metrics import all_forecasting_metrics

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results/tables"
WINDOW_SIZE = 7
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs(RESULTS_DIR, exist_ok=True)


def moving_average_predict(series, window=7):
    """Predict each value as the mean of the previous `window` actual values.

    For the first `window` positions there is not enough history; those are
    predicted as the mean of whatever history is available (at least 1 value).

    Args:
        series: 1-D numpy array of actual sales values (train + test concatenated).
        window: Number of past days to average.

    Returns:
        numpy array of predictions aligned with the original series.
    """
    predictions = []
    for i in range(len(series)):
        start = max(0, i - window)
        predictions.append(np.mean(series[start:i]) if i > 0 else series[0])
    return np.array(predictions)


def measure_inference_time(series, window=7, n_runs=1000):
    """Time 1000 single-step predictions and return average milliseconds.

    Args:
        series: Array used for timing test.
        window: Moving average window size.
        n_runs: Number of repetitions for timing.

    Returns:
        Average time per prediction in milliseconds.
    """
    start = time.perf_counter()
    for _ in range(n_runs):
        idx = np.random.randint(window, len(series))
        _ = np.mean(series[max(0, idx - window):idx])
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms


def main():
    print("=" * 60)
    print("STEP 2: Moving Average Baseline")
    print("=" * 60)

    train_path = os.path.join(PROCESSED_DIR, "train_data.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_data.csv")
    meta_path = os.path.join(PROCESSED_DIR, "split_meta.json")

    for p in [train_path, test_path, meta_path]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing file: {p}")
            print("  Run data_preprocessing.py first.")
            sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    families = meta["target_columns"]

    print(f"\n  Loading data for families: {families}")
    train_df = pd.read_csv(train_path, parse_dates=["date"])
    test_df = pd.read_csv(test_path, parse_dates=["date"])

    all_results = []

    for family in families:
        print(f"\n  --- {family} ---")
        train_series = train_df[family].values
        test_series = test_df[family].values

        # Concatenate to use actual train history when predicting first test points
        full_series = np.concatenate([train_series, test_series])
        all_preds = moving_average_predict(full_series, window=WINDOW_SIZE)

        # Extract predictions corresponding to the test period
        test_start_idx = len(train_series)
        y_pred = all_preds[test_start_idx:]
        y_true = test_series

        metrics = all_forecasting_metrics(y_true, y_pred)
        inf_time = measure_inference_time(full_series, window=WINDOW_SIZE)

        print(f"  MAPE: {metrics['MAPE (%)']:.2f}%  RMSE: {metrics['RMSE']:.4f}  MAE: {metrics['MAE']:.4f}")
        print(f"  Avg inference time: {inf_time:.4f} ms")

        row = {
            "Model": "Moving Average",
            "Family": family,
            **metrics,
            "Training_Time_s": 0.0,
            "Inference_Time_ms": round(inf_time, 4),
            "Model_Size_KB": 0.0  # No stored model
        }
        all_results.append(row)

        # Save per-family predictions
        pred_df = pd.DataFrame({
            "date": test_df["date"].values,
            "actual": y_true,
            "predicted": y_pred
        })
        pred_df.to_csv(
            f"{RESULTS_DIR}/forecast_moving_avg_{family.replace(' ', '_')}.csv",
            index=False
        )

    # Aggregate metrics across families (macro average)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{RESULTS_DIR}/forecast_moving_avg.csv", index=False)

    # Compute family-averaged metrics for the master comparison table
    avg = results_df[["MAPE (%)", "RMSE", "MAE"]].mean()
    print("\n  === Averaged Across Families ===")
    print(f"  MAPE: {avg['MAPE (%)']:.2f}%  RMSE: {avg['RMSE']:.4f}  MAE: {avg['MAE']:.4f}")

    # Save a single-row summary for the master comparison table
    summary = {
        "Model": "Moving Average (7-day)",
        "MAPE (%)": round(avg["MAPE (%)"], 2),
        "RMSE": round(avg["RMSE"], 4),
        "MAE": round(avg["MAE"], 4),
        "Training_Time_s": 0.0,
        "Inference_Time_ms": round(results_df["Inference_Time_ms"].mean(), 4),
        "Model_Size_KB": 0.0
    }
    pd.DataFrame([summary]).to_csv(
        f"{RESULTS_DIR}/forecast_moving_avg_summary.csv", index=False
    )

    print(f"\n  Saved: {RESULTS_DIR}/forecast_moving_avg.csv")
    print("  [DONE] Moving Average complete.")


if __name__ == "__main__":
    main()
