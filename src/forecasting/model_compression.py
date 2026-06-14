"""
Edge deployment compression for LSTM and CNN forecasting models.

Creates three versions of each model:
  1. Full .h5 model
  2. TFLite with dynamic range quantization
  3. TFLite with float16 quantization

Measures size, MAPE/RMSE/MAE, and inference time for each version.

Usage:
    python src/forecasting/model_compression.py
    (Requires model_lstm.py and model_cnn.py to have been run first.)
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
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    print("[ERROR] TensorFlow not installed. Run: pip install tensorflow>=2.12")
    sys.exit(1)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results/tables"
MODELS_DIR = "results/models"
WINDOW_SIZE = 14
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_sequences(data, window_size):
    """Reconstruct sliding-window sequences from a scaled feature array.

    Args:
        data: 2-D numpy array of shape (timesteps, features).
        window_size: Number of past timesteps to use as input.

    Returns:
        X: Shape (n_samples, window_size, features).
        y: Shape (n_samples,) — next-step first-column value.
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)


def convert_to_tflite_dynamic(model):
    """Apply dynamic range quantization and return TFLite model bytes.

    Args:
        model: Loaded Keras model.

    Returns:
        bytes of the converted TFLite model.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def convert_to_tflite_float16(model):
    """Apply float16 quantization and return TFLite model bytes.

    Args:
        model: Loaded Keras model.

    Returns:
        bytes of the converted TFLite model.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def run_tflite_inference(tflite_bytes, X_test):
    """Run inference using the TFLite interpreter.

    Args:
        tflite_bytes: Raw TFLite model bytes.
        X_test: Input array of shape (n_samples, window_size, features).

    Returns:
        Tuple (predictions array, avg_inference_time_ms).
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    times = []
    for sample in X_test:
        inp = np.expand_dims(sample, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], inp)
        t0 = time.perf_counter()
        interpreter.invoke()
        times.append(time.perf_counter() - t0)
        out = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(out[0][0])

    avg_ms = np.mean(times) * 1000
    return np.array(predictions), avg_ms


def process_model(arch_name, family, train_df, test_df, feature_cols):
    """Run full compression pipeline for one arch × family combination.

    Args:
        arch_name: 'lstm' or 'cnn'.
        family: Product family name string.
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        feature_cols: List of feature column names (excluding sales).

    Returns:
        List of result dicts (one per compression version).
    """
    safe_fam = family.replace(' ', '_')
    model_path = os.path.join(MODELS_DIR, f"{arch_name}_{safe_fam}.h5")
    scaler_path = os.path.join(MODELS_DIR, f"{arch_name}_scaler_{safe_fam}.pkl")

    if not os.path.exists(model_path):
        print(f"  [SKIP] {model_path} not found — run model_{arch_name}.py first.")
        return []

    model = tf.keras.models.load_model(model_path, compile=False)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    cols = [family] + feature_cols
    train_data = train_df[cols].values.astype(float)
    test_data = test_df[cols].values.astype(float)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    full_scaled = np.concatenate([train_scaled, test_scaled], axis=0)
    test_start = len(train_scaled)
    X_test, y_test_scaled = build_sequences(full_scaled[test_start - WINDOW_SIZE:], WINDOW_SIZE)

    n_features = X_test.shape[2]
    dummy_true = np.zeros((len(y_test_scaled), n_features))
    dummy_true[:, 0] = y_test_scaled
    y_true = scaler.inverse_transform(dummy_true)[:, 0]

    results = []
    full_size_kb = os.path.getsize(model_path) / 1024

    versions = [
        ("Full (.h5)", None),
        ("TFLite Dynamic Quant", "dynamic"),
        ("TFLite Float16 Quant", "float16"),
    ]

    for version_name, quant_type in versions:
        print(f"    [{arch_name.upper()} {family}] {version_name}")

        if quant_type is None:
            # Full model inference
            t0 = time.perf_counter()
            y_pred_scaled = model.predict(X_test, verbose=0)
            inf_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000
            dummy = np.zeros((len(y_pred_scaled), n_features))
            dummy[:, 0] = y_pred_scaled.flatten()
            y_pred = scaler.inverse_transform(dummy)[:, 0]
            size_kb = full_size_kb
            compression_ratio = 1.0
        else:
            if quant_type == "dynamic":
                tflite_bytes = convert_to_tflite_dynamic(model)
            else:
                tflite_bytes = convert_to_tflite_float16(model)

            tflite_path = os.path.join(
                MODELS_DIR,
                f"{arch_name}_{safe_fam}_{quant_type}.tflite"
            )
            with open(tflite_path, "wb") as f:
                f.write(tflite_bytes)
            size_kb = os.path.getsize(tflite_path) / 1024
            compression_ratio = round(full_size_kb / size_kb, 2)

            y_pred_scaled_flat, inf_ms = run_tflite_inference(tflite_bytes, X_test)
            dummy = np.zeros((len(y_pred_scaled_flat), n_features))
            dummy[:, 0] = y_pred_scaled_flat
            y_pred = scaler.inverse_transform(dummy)[:, 0]

        metrics = all_forecasting_metrics(y_true, y_pred)
        print(f"      Size: {size_kb:.2f} KB  MAPE: {metrics['MAPE (%)']:.2f}%  "
              f"Inference: {inf_ms:.4f} ms  Ratio: {compression_ratio}x")

        results.append({
            "Architecture": arch_name.upper(),
            "Family": family,
            "Version": version_name,
            "Size_KB": round(size_kb, 2),
            "MAPE (%)": metrics["MAPE (%)"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "Inference_Time_ms": round(inf_ms, 4),
            "Compression_Ratio": compression_ratio
        })

    return results


def main():
    print("=" * 60)
    print("STEP 6: Model Compression for Edge Deployment")
    print("=" * 60)

    meta_path = os.path.join(PROCESSED_DIR, "split_meta.json")
    train_path = os.path.join(PROCESSED_DIR, "train_data.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_data.csv")

    for p in [meta_path, train_path, test_path]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing file: {p}")
            sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    families = meta["target_columns"]
    feature_cols = meta["feature_columns"]

    train_df = pd.read_csv(train_path, parse_dates=["date"])
    test_df = pd.read_csv(test_path, parse_dates=["date"])

    all_results = []
    for arch in ["lstm", "cnn"]:
        print(f"\n  === {arch.upper()} Compression ===")
        for family in families:
            rows = process_model(arch, family, train_df, test_df, feature_cols)
            all_results.extend(rows)

    results_df = pd.DataFrame(all_results)
    out_path = f"{RESULTS_DIR}/compression_comparison.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\n  Saved: {out_path}")
    print("\n  === Compression Summary (avg across families) ===")
    summary = results_df.groupby(["Architecture", "Version"])[
        ["Size_KB", "MAPE (%)", "Inference_Time_ms", "Compression_Ratio"]
    ].mean().round(3)
    print(summary.to_string())
    print("\n  [DONE] Model compression complete.")


if __name__ == "__main__":
    main()
