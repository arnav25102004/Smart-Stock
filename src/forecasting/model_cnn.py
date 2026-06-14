"""
1D Convolutional Neural Network for demand forecasting.

Uses the same 14-day sliding window and MinMaxScaler setup as model_lstm.py
for a fair apples-to-apples comparison.

Usage:
    python src/forecasting/model_cnn.py
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
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    print("[ERROR] TensorFlow not installed. Run: pip install tensorflow>=2.12")
    sys.exit(1)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results/tables"
MODELS_DIR = "results/models"
WINDOW_SIZE = 14
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 10
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def build_sequences(data, window_size):
    """Convert a 2-D time series array into (X, y) sliding-window sequences.

    Args:
        data: 2-D numpy array of shape (timesteps, features).
        window_size: Number of past timesteps used as input.

    Returns:
        X: Shape (n_samples, window_size, features).
        y: Shape (n_samples,) — next-step value for the first feature (sales).
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)


def build_cnn_model(window_size, n_features):
    """Construct the two-block 1D-CNN architecture.

    Args:
        window_size: Number of timesteps in each input sequence.
        n_features: Number of input features per timestep.

    Returns:
        Compiled Keras Sequential model.
    """
    model = keras.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(50, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse'
    )
    return model


def main():
    print("=" * 60)
    print("STEP 5: Training CNN Model")
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
    feature_cols = meta["feature_columns"]

    train_df = pd.read_csv(train_path, parse_dates=["date"])
    test_df = pd.read_csv(test_path, parse_dates=["date"])

    all_results = []

    for family in families:
        print(f"\n  --- {family} ---")

        cols = [family] + feature_cols
        train_data = train_df[cols].values.astype(float)
        test_data = test_df[cols].values.astype(float)

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        full_scaled = np.concatenate([train_scaled, test_scaled], axis=0)
        X_train, y_train = build_sequences(train_scaled, WINDOW_SIZE)
        test_start = len(train_scaled)
        X_test, y_test = build_sequences(full_scaled[test_start - WINDOW_SIZE:], WINDOW_SIZE)

        n_features = X_train.shape[2]
        model = build_cnn_model(WINDOW_SIZE, n_features)
        print(f"  Parameters: {model.count_params():,}")

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE, restore_best_weights=True
        )

        print(f"  Training (max {EPOCHS} epochs, early stopping patience={PATIENCE})...")
        t0 = time.perf_counter()
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )
        train_time = time.perf_counter() - t0
        actual_epochs = len(history.history['loss'])
        print(f"  Training time: {train_time:.2f}s  (stopped at epoch {actual_epochs})")

        model_path = os.path.join(MODELS_DIR, f"cnn_{family.replace(' ', '_')}.h5")
        model.save(model_path)
        model_size_kb = os.path.getsize(model_path) / 1024

        scaler_path = os.path.join(MODELS_DIR, f"cnn_scaler_{family.replace(' ', '_')}.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        t0 = time.perf_counter()
        y_pred_scaled = model.predict(X_test, verbose=0)
        inf_time_total = time.perf_counter() - t0
        inf_time_ms = (inf_time_total / len(X_test)) * 1000

        dummy = np.zeros((len(y_pred_scaled), n_features))
        dummy[:, 0] = y_pred_scaled.flatten()
        y_pred = scaler.inverse_transform(dummy)[:, 0]

        dummy_true = np.zeros((len(y_test), n_features))
        dummy_true[:, 0] = y_test
        y_true = scaler.inverse_transform(dummy_true)[:, 0]

        metrics = all_forecasting_metrics(y_true, y_pred)
        print(f"  MAPE: {metrics['MAPE (%)']:.2f}%  RMSE: {metrics['RMSE']:.4f}  MAE: {metrics['MAE']:.4f}")
        print(f"  Model size: {model_size_kb:.2f} KB  Inference: {inf_time_ms:.4f} ms/sample")

        history_df = pd.DataFrame({
            "epoch": range(1, actual_epochs + 1),
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"]
        })
        history_df.to_csv(
            f"{RESULTS_DIR}/cnn_history_{family.replace(' ', '_')}.csv", index=False
        )

        row = {
            "Model": "CNN",
            "Family": family,
            **metrics,
            "Training_Time_s": round(train_time, 2),
            "Inference_Time_ms": round(inf_time_ms, 4),
            "Model_Size_KB": round(model_size_kb, 2),
            "Num_Parameters": model.count_params()
        }
        all_results.append(row)

        pred_df = pd.DataFrame({
            "date": test_df["date"].values[:len(y_true)],
            "actual": y_true,
            "predicted": y_pred
        })
        pred_df.to_csv(
            f"{RESULTS_DIR}/forecast_cnn_{family.replace(' ', '_')}.csv",
            index=False
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{RESULTS_DIR}/forecast_cnn.csv", index=False)

    avg = results_df[["MAPE (%)", "RMSE", "MAE"]].mean()
    print("\n  === Averaged Across Families ===")
    print(f"  MAPE: {avg['MAPE (%)']:.2f}%  RMSE: {avg['RMSE']:.4f}  MAE: {avg['MAE']:.4f}")

    summary = {
        "Model": "CNN",
        "MAPE (%)": round(avg["MAPE (%)"], 2),
        "RMSE": round(avg["RMSE"], 4),
        "MAE": round(avg["MAE"], 4),
        "Training_Time_s": round(results_df["Training_Time_s"].sum(), 2),
        "Inference_Time_ms": round(results_df["Inference_Time_ms"].mean(), 4),
        "Model_Size_KB": round(results_df["Model_Size_KB"].mean(), 2)
    }
    pd.DataFrame([summary]).to_csv(
        f"{RESULTS_DIR}/forecast_cnn_summary.csv", index=False
    )

    print(f"\n  Saved: {RESULTS_DIR}/forecast_cnn.csv")
    print("  [DONE] CNN model complete.")


if __name__ == "__main__":
    main()
