"""
AT-CNN: Attention-augmented CNN-LSTM hybrid for demand forecasting.

Architecture:
  1. CNN Block  — two Conv1D layers with BatchNorm to extract local patterns
  2. Channel Attention — soft-weights the 64 CNN feature maps
  3. LSTM Block — captures temporal dependencies across the attended features
  4. Dense Output — projects to a single sales value

Trained twice per product family:
  - Version A  : standard MSE loss
  - Version B  : Retail-Aware Asymmetric Loss (RAL, alpha=3.0) that penalises
                 under-prediction (stockout risk) 3× more than over-prediction.

New metric:
  Stockout Risk Rate (SRR) — % of test samples where the model under-predicted
  by more than 10%. Lower is better.

Usage:
    python src/forecasting/model_atcnn.py
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Constants (identical to model_lstm.py / model_cnn.py) ─────────────────────
PROCESSED_DIR = "data/processed"
RESULTS_DIR   = "results/tables"
MODELS_DIR    = "results/models"
PLOTS_DIR     = "results/plots"
WINDOW_SIZE   = 14
EPOCHS        = 100
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
PATIENCE      = 10
RANDOM_SEED   = 42
EDGE_SIZE_KB_THRESHOLD = 500   # models below this are "edge deployable"

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
for d in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Loss functions ─────────────────────────────────────────────────────────────

def retail_asymmetric_loss(alpha=3.0):
    """
    Retail-Aware Asymmetric Loss (RAL).
    Penalises under-prediction (stockout risk) alpha times more than
    over-prediction (overstock).
    """
    def loss(y_true, y_pred):
        error = y_true - y_pred
        under = tf.maximum(error, 0.0)   # y_pred < y_true  → stockout
        over  = tf.maximum(-error, 0.0)  # y_pred > y_true  → overstock
        return tf.reduce_mean(alpha * tf.square(under) + tf.square(over))
    return loss


# ── Metric ─────────────────────────────────────────────────────────────────────

def stockout_risk_rate(y_true, y_pred, threshold=0.9):
    """
    Percentage of predictions where the model under-predicted by more than
    (1 - threshold)*100 percent of the true value.
    Lower is better.
    """
    count = sum(1 for t, p in zip(y_true, y_pred) if t > 0 and p < t * threshold)
    return round((count / len(y_true)) * 100, 2)


# ── Data helpers ───────────────────────────────────────────────────────────────

def build_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)


# ── Model builder ──────────────────────────────────────────────────────────────

def build_atcnn_model(window_size, n_features, loss_fn='mse'):
    """
    AT-CNN: CNN → Channel Attention → LSTM → Dense.

    Args:
        window_size: Length of the input sequence.
        n_features:  Number of features per timestep.
        loss_fn:     Keras loss name or callable (e.g. retail_asymmetric_loss()).

    Returns:
        Compiled Keras Model.
    """
    inp = keras.Input(shape=(window_size, n_features))

    # ── Step 1: CNN Block ──────────────────────────────────────────────────────
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)   # shape: (batch, window_size, 64)

    # ── Step 2: Channel Attention ──────────────────────────────────────────────
    attn = layers.Dense(64, activation='tanh')(x)
    attn = layers.Dense(64, activation='softmax')(attn)
    x    = layers.Multiply()([x, attn])  # shape: (batch, window_size, 64)

    # ── Step 3: LSTM Block ─────────────────────────────────────────────────────
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)

    # ── Step 4: Output ─────────────────────────────────────────────────────────
    x   = layers.Dense(25, activation='relu')(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn
    )
    return model


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _save_loss_curve(history, model_tag, family_safe):
    epochs_ran = len(history.history['loss'])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, epochs_ran + 1), history.history['loss'],
            label='Training Loss', color='#2ecc71', linewidth=1.5)
    ax.plot(range(1, epochs_ran + 1), history.history['val_loss'],
            label='Validation Loss', color='#e74c3c', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Convergence — {model_tag} ({family_safe})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fname = f"{PLOTS_DIR}/loss_atcnn_{model_tag}_{family_safe}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def _save_actual_vs_pred(y_true, y_pred, model_tag, family, family_safe):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true, label='Actual',    color='#1f77b4', linewidth=1.5)
    ax.plot(y_pred, label='Predicted', color='#ff7f0e', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Sales (Units)')
    ax.set_title(f'Actual vs Predicted — {model_tag} ({family})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fname = f"{PLOTS_DIR}/actual_vs_pred_atcnn_{model_tag}_{family_safe}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def _save_mape_srr_scatter(rows):
    """MAPE vs Stockout Risk Rate scatter for all models (all families averaged)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']
    for i, r in enumerate(rows):
        ax.scatter(r['SRR (%)'], r['MAPE (%)'],
                   s=140, c=colors[i % len(colors)],
                   edgecolors='black', linewidth=0.6, zorder=5)
        ax.annotate(r['Model'], (r['SRR (%)'], r['MAPE (%)']),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)

    ax.set_xlabel('Stockout Risk Rate — SRR (%)', fontsize=12)
    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_title('MAPE vs Stockout Risk Rate\n(lower-left = best position)', fontsize=13)
    ax.grid(True, alpha=0.3)

    # annotate best quadrant
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.annotate('← Best region\n(low MAPE + low SRR)',
                xy=(xlim[0] + (xlim[1] - xlim[0]) * 0.03,
                    ylim[0] + (ylim[1] - ylim[0]) * 0.05),
                fontsize=8, color='green', style='italic')

    plt.savefig(f"{PLOTS_DIR}/mape_vs_srr_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()


# ── Per-family training ────────────────────────────────────────────────────────

def train_one_family(family, feature_cols, train_df, test_df, loss_variant):
    """
    Train and evaluate AT-CNN for one product family and one loss variant.

    Args:
        family:       Product family string (e.g. 'GROCERY I').
        feature_cols: List of additional feature column names.
        train_df:     Training DataFrame.
        test_df:      Test DataFrame.
        loss_variant: 'mse' or 'ral'.

    Returns:
        Dict with metrics, plus numpy arrays y_true, y_pred, and history.
    """
    cols       = [family] + feature_cols
    train_data = train_df[cols].values.astype(float)
    test_data  = test_df[cols].values.astype(float)

    scaler      = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled  = scaler.transform(test_data)

    full_scaled = np.concatenate([train_scaled, test_scaled], axis=0)
    X_train, y_train = build_sequences(train_scaled, WINDOW_SIZE)
    test_start        = len(train_scaled)
    X_test,  y_test  = build_sequences(full_scaled[test_start - WINDOW_SIZE:], WINDOW_SIZE)

    n_features = X_train.shape[2]

    loss_fn   = retail_asymmetric_loss(alpha=3.0) if loss_variant == 'ral' else 'mse'
    model_tag = f"MSE" if loss_variant == 'mse' else f"RAL"

    model = build_atcnn_model(WINDOW_SIZE, n_features, loss_fn=loss_fn)

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE, restore_best_weights=True
    )

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

    family_safe = family.replace(' ', '_')

    # Save model
    model_fname = f"atcnn_{loss_variant}_{family_safe}.h5"
    model_path  = os.path.join(MODELS_DIR, model_fname)
    model.save(model_path)
    model_size_kb = os.path.getsize(model_path) / 1024

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, f"atcnn_{loss_variant}_scaler_{family_safe}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Inference
    t0 = time.perf_counter()
    y_pred_scaled = model.predict(X_test, verbose=0)
    inf_time_ms   = ((time.perf_counter() - t0) / len(X_test)) * 1000

    # Inverse-transform
    dummy       = np.zeros((len(y_pred_scaled), n_features))
    dummy[:, 0] = y_pred_scaled.flatten()
    y_pred      = scaler.inverse_transform(dummy)[:, 0]

    dummy_true       = np.zeros((len(y_test), n_features))
    dummy_true[:, 0] = y_test
    y_true           = scaler.inverse_transform(dummy_true)[:, 0]

    metrics = all_forecasting_metrics(y_true, y_pred)
    srr     = stockout_risk_rate(y_true, y_pred)

    # Save training history CSV
    hist_df = pd.DataFrame({
        "epoch":      range(1, actual_epochs + 1),
        "train_loss": history.history["loss"],
        "val_loss":   history.history["val_loss"]
    })
    hist_df.to_csv(f"{RESULTS_DIR}/atcnn_{loss_variant}_history_{family_safe}.csv", index=False)

    # Plots
    _save_loss_curve(history, model_tag, family_safe)
    _save_actual_vs_pred(y_true, y_pred, model_tag, family, family_safe)

    # Per-family predictions CSV
    pred_df = pd.DataFrame({
        "date":      test_df["date"].values[:len(y_true)],
        "actual":    y_true,
        "predicted": y_pred
    })
    pred_df.to_csv(
        f"{RESULTS_DIR}/forecast_atcnn_{loss_variant}_{family_safe}.csv", index=False
    )

    return {
        "family":        family,
        "loss_variant":  loss_variant,
        "model_tag":     model_tag,
        "y_true":        y_true,
        "y_pred":        y_pred,
        "metrics":       metrics,
        "srr":           srr,
        "train_time":    round(train_time, 2),
        "inf_time_ms":   round(inf_time_ms, 4),
        "model_size_kb": round(model_size_kb, 2),
        "num_params":    model.count_params(),
        "actual_epochs": actual_epochs
    }


# ── Load prior-model SRR for full comparison table ─────────────────────────────

def compute_srr_for_prior_models(families, feature_cols, test_df):
    """
    Read prediction CSVs for Moving Average, Prophet, LSTM, CNN and compute SRR.
    Returns a dict: {model_label: avg_srr}.
    """
    prior = {
        "Moving Average": "forecast_moving_avg",
        "Prophet":        "forecast_prophet",
        "LSTM":           "forecast_lstm",
        "CNN":            "forecast_cnn"
    }
    srr_map = {}
    for label, prefix in prior.items():
        srrs = []
        for fam in families:
            safe = fam.replace(' ', '_')
            path = os.path.join(RESULTS_DIR, f"{prefix}_{safe}.csv")
            if not os.path.exists(path):
                continue
            df   = pd.read_csv(path)
            srrs.append(stockout_risk_rate(df["actual"].values, df["predicted"].values))
        srr_map[label] = round(np.mean(srrs), 2) if srrs else float('nan')
    return srr_map


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("AT-CNN: Attention-augmented CNN-LSTM Demand Forecasting")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────────
    train_path = os.path.join(PROCESSED_DIR, "train_data.csv")
    test_path  = os.path.join(PROCESSED_DIR, "test_data.csv")
    meta_path  = os.path.join(PROCESSED_DIR, "split_meta.json")

    for p in [train_path, test_path, meta_path]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing file: {p}")
            print("  Run data_preprocessing.py first.")
            sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    families     = meta["target_columns"]
    feature_cols = meta["feature_columns"]

    train_df = pd.read_csv(train_path, parse_dates=["date"])
    test_df  = pd.read_csv(test_path,  parse_dates=["date"])

    # ── Train MSE and RAL versions ─────────────────────────────────────────────
    all_rows_mse = []
    all_rows_ral = []

    for family in families:
        family_safe = family.replace(' ', '_')
        print(f"\n  -- {family} --")

        for variant, bucket in [('mse', all_rows_mse), ('ral', all_rows_ral)]:
            tag = "MSE" if variant == 'mse' else "RAL"
            print(f"    Training AT-CNN ({tag})...")
            result = train_one_family(family, feature_cols, train_df, test_df, variant)
            m = result['metrics']
            print(f"      Epochs: {result['actual_epochs']}  "
                  f"Time: {result['train_time']:.1f}s  "
                  f"Size: {result['model_size_kb']:.1f} KB")
            print(f"      MAPE: {m['MAPE (%)']:.2f}%  "
                  f"RMSE: {m['RMSE']:.4f}  "
                  f"MAE: {m['MAE']:.4f}  "
                  f"SRR: {result['srr']:.2f}%")
            bucket.append(result)

    # ── Aggregate per-variant ──────────────────────────────────────────────────
    def aggregate(rows, model_label):
        mapes  = [r['metrics']['MAPE (%)'] for r in rows]
        rmses  = [r['metrics']['RMSE']     for r in rows]
        maes   = [r['metrics']['MAE']      for r in rows]
        srrs   = [r['srr']                 for r in rows]
        sizes  = [r['model_size_kb']       for r in rows]
        inf_ms = [r['inf_time_ms']         for r in rows]
        return {
            "Model":              model_label,
            "Loss":               rows[0]['loss_variant'].upper(),
            "MAPE (%)":           round(np.mean(mapes), 2),
            "RMSE":               round(np.mean(rmses), 4),
            "MAE":                round(np.mean(maes),  4),
            "SRR (%)":            round(np.mean(srrs),  2),
            "Training_Time_s":    round(sum(r['train_time'] for r in rows), 2),
            "Inference_Time_ms":  round(np.mean(inf_ms), 4),
            "Model_Size_KB":      round(np.mean(sizes), 2),
            "Num_Parameters":     rows[0]['num_params']
        }

    agg_mse = aggregate(all_rows_mse, "AT-CNN (MSE)")
    agg_ral = aggregate(all_rows_ral, "AT-CNN (RAL)")

    # ── Save AT-CNN result tables ──────────────────────────────────────────────
    atcnn_rows_full = []
    for r in all_rows_mse + all_rows_ral:
        atcnn_rows_full.append({
            "Model":             f"AT-CNN ({r['loss_variant'].upper()})",
            "Family":            r['family'],
            **r['metrics'],
            "SRR (%)":           r['srr'],
            "Training_Time_s":   r['train_time'],
            "Inference_Time_ms": r['inf_time_ms'],
            "Model_Size_KB":     r['model_size_kb'],
            "Num_Parameters":    r['num_params']
        })
    pd.DataFrame(atcnn_rows_full).to_csv(f"{RESULTS_DIR}/atcnn_results.csv", index=False)
    print(f"\n  Saved: {RESULTS_DIR}/atcnn_results.csv")

    # Summary for evaluate.py
    for agg, suffix in [(agg_mse, 'mse'), (agg_ral, 'ral')]:
        summary = {k: v for k, v in agg.items()
                   if k not in ('Loss', 'SRR (%)', 'Num_Parameters')}
        pd.DataFrame([summary]).to_csv(
            f"{RESULTS_DIR}/forecast_atcnn_{suffix}_summary.csv", index=False
        )

    # ── Build full comparison table with SRR ──────────────────────────────────
    print("\n  Computing SRR for prior models...")
    prior_srr = compute_srr_for_prior_models(families, feature_cols, test_df)

    # Load prior summaries
    prior_summaries_raw = {
        "Moving Average": "forecast_moving_avg_summary.csv",
        "Prophet":        "forecast_prophet_summary.csv",
        "LSTM":           "forecast_lstm_summary.csv",
        "CNN":            "forecast_cnn_summary.csv",
    }
    full_table_rows = []
    for label, fname in prior_summaries_raw.items():
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            continue
        row = pd.read_csv(path).iloc[0].to_dict()
        full_table_rows.append({
            "Model":             label,
            "Loss":              "MSE",
            "MAPE (%)":          row.get("MAPE (%)", float('nan')),
            "RMSE":              row.get("RMSE",     float('nan')),
            "SRR (%)":           prior_srr.get(label, float('nan')),
            "Model_Size_KB":     row.get("Model_Size_KB", 0),
            "Inference_Time_ms": row.get("Inference_Time_ms", 0)
        })

    for agg in [agg_mse, agg_ral]:
        full_table_rows.append({
            "Model":             agg["Model"],
            "Loss":              agg["Loss"],
            "MAPE (%)":          agg["MAPE (%)"],
            "RMSE":              agg["RMSE"],
            "SRR (%)":           agg["SRR (%)"],
            "Model_Size_KB":     agg["Model_Size_KB"],
            "Inference_Time_ms": agg["Inference_Time_ms"]
        })

    full_df = pd.DataFrame(full_table_rows)
    full_df.to_csv(f"{RESULTS_DIR}/full_comparison_with_ral.csv", index=False)
    print(f"  Saved: {RESULTS_DIR}/full_comparison_with_ral.csv")

    # ── MAPE vs SRR scatter ────────────────────────────────────────────────────
    scatter_rows = [
        {"Model": r["Model"], "MAPE (%)": r["MAPE (%)"], "SRR (%)": r["SRR (%)"]}
        for r in full_table_rows
    ]
    _save_mape_srr_scatter(scatter_rows)
    print(f"  Saved: {PLOTS_DIR}/mape_vs_srr_scatter.png")

    # ── Update forecasting_comparison.csv with AT-CNN (MSE) row ───────────────
    comp_path = os.path.join(RESULTS_DIR, "forecasting_comparison.csv")
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        # Remove any stale AT-CNN entries and re-append
        comp_df = comp_df[~comp_df["Model"].str.startswith("AT-CNN")]
        new_row = {
            "Model":             agg_mse["Model"],
            "MAPE (%)":          agg_mse["MAPE (%)"],
            "RMSE":              agg_mse["RMSE"],
            "MAE":               agg_mse["MAE"],
            "Training_Time_s":   agg_mse["Training_Time_s"],
            "Inference_Time_ms": agg_mse["Inference_Time_ms"],
            "Model_Size_KB":     agg_mse["Model_Size_KB"]
        }
        comp_df = pd.concat([comp_df, pd.DataFrame([new_row])], ignore_index=True)
        comp_df.to_csv(comp_path, index=False)
        print(f"  Updated: {comp_path}")

    # ── Print summary comparisons ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FULL COMPARISON TABLE (averaged across product families)")
    print("=" * 65)
    print(full_df.to_string(index=False))

    # CNN baseline
    cnn_path = os.path.join(RESULTS_DIR, "forecast_cnn_summary.csv")
    lstm_path = os.path.join(RESULTS_DIR, "forecast_lstm_summary.csv")
    if os.path.exists(cnn_path) and os.path.exists(lstm_path):
        cnn_mape  = pd.read_csv(cnn_path).iloc[0]["MAPE (%)"]
        lstm_mape = pd.read_csv(lstm_path).iloc[0]["MAPE (%)"]

        atcnn_mape = agg_mse["MAPE (%)"]
        if cnn_mape > 0:
            cnn_improvement  = round((cnn_mape  - atcnn_mape) / cnn_mape  * 100, 1)
        else:
            cnn_improvement  = float('nan')
        if lstm_mape > 0:
            lstm_improvement = round((lstm_mape - atcnn_mape) / lstm_mape * 100, 1)
        else:
            lstm_improvement = float('nan')

        size_kb    = agg_mse["Model_Size_KB"]
        edge_label = "YES" if size_kb < EDGE_SIZE_KB_THRESHOLD else "NO"

        print(f"\n  AT-CNN vs CNN  improvement : {cnn_improvement:+.1f}% MAPE")
        print(f"  AT-CNN vs LSTM improvement : {lstm_improvement:+.1f}% MAPE")
        print(f"  AT-CNN size                : {size_kb:.1f} KB  (edge deployable: {edge_label})")
        print(f"\n  AT-CNN (RAL) SRR           : {agg_ral['SRR (%)']:.2f}%")
        print(f"  AT-CNN (MSE) SRR           : {agg_mse['SRR (%)']:.2f}%")
        ral_srr_improvement = round(agg_mse['SRR (%)'] - agg_ral['SRR (%)'], 2)
        print(f"  RAL stockout improvement   : {ral_srr_improvement:+.2f} pp vs MSE")

    print("\n  [DONE] AT-CNN training and evaluation complete.")


if __name__ == "__main__":
    main()
