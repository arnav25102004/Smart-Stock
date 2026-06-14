"""
Facebook Prophet demand forecasting model.

Trains one Prophet model per product family using the chronological train/test
split from data_preprocessing.py. Records training time, prediction time, and
model file size (pickled).

Usage:
    python src/forecasting/model_prophet.py
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.metrics import all_forecasting_metrics

try:
    from prophet import Prophet
except ImportError:
    print("[ERROR] Prophet not installed. Run: pip install prophet")
    sys.exit(1)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results/tables"
MODELS_DIR = "results/models"
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def train_prophet(train_df, family):
    """Train a Prophet model on one product family.

    Args:
        train_df: DataFrame with 'date' and family sales columns.
        family: String name of the product family to model.

    Returns:
        Fitted Prophet model.
    """
    prophet_df = train_df[["date", family]].copy()
    prophet_df = prophet_df.rename(columns={"date": "ds", family: "y"})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95
    )
    model.fit(prophet_df)
    return model


def predict_prophet(model, test_df):
    """Run Prophet inference on the test period.

    Args:
        model: Fitted Prophet model.
        test_df: DataFrame with a 'date' column covering the test period.

    Returns:
        numpy array of predicted 'yhat' values aligned with test_df rows.
    """
    future = pd.DataFrame({"ds": test_df["date"].values})
    forecast = model.predict(future)
    return forecast["yhat"].values


def main():
    print("=" * 60)
    print("STEP 3: Training Prophet Model")
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

    train_df = pd.read_csv(train_path, parse_dates=["date"])
    test_df = pd.read_csv(test_path, parse_dates=["date"])

    all_results = []

    for family in families:
        print(f"\n  --- {family} ---")

        # Training
        print(f"  Training Prophet for {family}...")
        t0 = time.perf_counter()
        model = train_prophet(train_df, family)
        train_time = time.perf_counter() - t0
        print(f"  Training time: {train_time:.2f}s")

        # Save model and measure size
        model_path = os.path.join(MODELS_DIR, f"prophet_{family.replace(' ', '_')}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        model_size_kb = os.path.getsize(model_path) / 1024

        # Prediction + timing
        t0 = time.perf_counter()
        y_pred = predict_prophet(model, test_df)
        pred_time_total = time.perf_counter() - t0
        inf_time_ms = (pred_time_total / len(test_df)) * 1000
        print(f"  Inference time (per sample): {inf_time_ms:.4f} ms")

        y_true = test_df[family].values
        metrics = all_forecasting_metrics(y_true, y_pred)
        print(f"  MAPE: {metrics['MAPE (%)']:.2f}%  RMSE: {metrics['RMSE']:.4f}  MAE: {metrics['MAE']:.4f}")
        print(f"  Model size: {model_size_kb:.2f} KB")

        row = {
            "Model": "Prophet",
            "Family": family,
            **metrics,
            "Training_Time_s": round(train_time, 2),
            "Inference_Time_ms": round(inf_time_ms, 4),
            "Model_Size_KB": round(model_size_kb, 2)
        }
        all_results.append(row)

        pred_df = pd.DataFrame({
            "date": test_df["date"].values,
            "actual": y_true,
            "predicted": y_pred
        })
        pred_df.to_csv(
            f"{RESULTS_DIR}/forecast_prophet_{family.replace(' ', '_')}.csv",
            index=False
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{RESULTS_DIR}/forecast_prophet.csv", index=False)

    avg = results_df[["MAPE (%)", "RMSE", "MAE"]].mean()
    print("\n  === Averaged Across Families ===")
    print(f"  MAPE: {avg['MAPE (%)']:.2f}%  RMSE: {avg['RMSE']:.4f}  MAE: {avg['MAE']:.4f}")

    summary = {
        "Model": "Prophet",
        "MAPE (%)": round(avg["MAPE (%)"], 2),
        "RMSE": round(avg["RMSE"], 4),
        "MAE": round(avg["MAE"], 4),
        "Training_Time_s": round(results_df["Training_Time_s"].sum(), 2),
        "Inference_Time_ms": round(results_df["Inference_Time_ms"].mean(), 4),
        "Model_Size_KB": round(results_df["Model_Size_KB"].mean(), 2)
    }
    pd.DataFrame([summary]).to_csv(
        f"{RESULTS_DIR}/forecast_prophet_summary.csv", index=False
    )

    print(f"\n  Saved: {RESULTS_DIR}/forecast_prophet.csv")
    print("  [DONE] Prophet model complete.")


if __name__ == "__main__":
    main()
