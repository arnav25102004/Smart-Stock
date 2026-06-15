"""
Master results generator for the SMARTSTOCK research paper.

Runs both experiment pipelines and compiles all five paper tables:
  Table 1 — Forecasting model comparison
  Table 2 — Forecasting model compression
  Table 3 — NLP intent classification comparison
  Table 4 — NLP model compression
  Table 5 — End-to-end pipeline performance

Usage:
    python src/generate_all_results.py
"""

import subprocess
import sys
import time
import os
import pandas as pd

RESULTS_DIR = "results/tables"


# ─────────────────────── Pipeline runner ─────────────────────────────────────

def run_pipeline(script, label):
    """Run a pipeline script as a subprocess.

    Args:
        script: Path to the Python script.
        label: Short description printed to console.

    Returns:
        True on success, False on failure.
    """
    print(f"\n{'=' * 60}")
    print(label)
    print("=" * 60)
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, script])
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"[ERROR] {script} failed.")
        return False
    print(f"  Done in {elapsed:.1f}s")
    return True


# ─────────────────────── Table builders ──────────────────────────────────────

def build_table1():
    """Table 1 — Forecasting Model Comparison (macro avg across families).

    Returns:
        DataFrame or None.
    """
    path = os.path.join(RESULTS_DIR, "forecasting_comparison.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    cols = ["Model", "MAPE (%)", "RMSE", "MAE",
            "Training_Time_s", "Inference_Time_ms", "Model_Size_KB"]
    available = [c for c in cols if c in df.columns]
    return df[available]


def build_table2():
    """Table 2 — Forecasting Model Compression for Edge.

    Returns:
        DataFrame or None.
    """
    path = os.path.join(RESULTS_DIR, "compression_comparison.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Average across families
    grp = df.groupby(["Architecture", "Version"]).agg({
        "Size_KB": "mean",
        "MAPE (%)": "mean",
        "RMSE": "mean",
        "Inference_Time_ms": "mean",
        "Compression_Ratio": "mean",
    }).round(3).reset_index()
    return grp


def build_table3():
    """Table 3 — NLP Intent Classification Comparison.

    Returns:
        DataFrame or None.
    """
    path = os.path.join(RESULTS_DIR, "nlp_comparison.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def build_table4():
    """Table 4 — NLP Model Compression for Edge.

    Returns:
        DataFrame or None.
    """
    path = os.path.join(RESULTS_DIR, "nlp_compression_comparison.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    cols = ["Version", "Size_KB", "Accuracy (%)", "Inference_Time_ms", "Compression_Ratio"]
    available = [c for c in cols if c in df.columns]
    return df[available]


def build_table5():
    """Table 5 — End-to-End Pipeline Performance.

    Returns:
        DataFrame or None.
    """
    summary_path = os.path.join(RESULTS_DIR, "pipeline_summary.csv")
    nlp_path = os.path.join(RESULTS_DIR, "nlp_comparison.csv")
    forecast_path = os.path.join(RESULTS_DIR, "forecasting_comparison.csv")

    rows = []

    if os.path.exists(summary_path):
        s = pd.read_csv(summary_path).iloc[0]
        rows.append({"Metric": "End-to-End Latency (ms)", "Value": s.get("Avg_Latency_ms", "N/A")})
        rows.append({"Metric": "NLP Model", "Value": s.get("NLP_Model", "N/A")})
        rows.append({"Metric": "Forecasting Model", "Value": s.get("Forecast_Model", "N/A")})

    if os.path.exists(nlp_path):
        nlp_df = pd.read_csv(nlp_path)
        best_nlp = nlp_df.sort_values("Overall Acc (%)", ascending=False).iloc[0]
        rows.append({"Metric": "Best NLP Accuracy (%)", "Value": best_nlp.get("Overall Acc (%)", "N/A")})
        rows.append({"Metric": "Best NLP F1 (macro)", "Value": best_nlp.get("F1 (macro)", "N/A")})

    if os.path.exists(forecast_path):
        fc_df = pd.read_csv(forecast_path)
        if "MAPE (%)" in fc_df.columns:
            best_fc = fc_df.sort_values("MAPE (%)").iloc[0]
            rows.append({"Metric": "Best Forecasting MAPE (%)", "Value": best_fc.get("MAPE (%)", "N/A")})
            rows.append({"Metric": "Best Forecasting Model", "Value": best_fc.get("Model", "N/A")})

    return pd.DataFrame(rows) if rows else None


def print_and_save_table(df, title, filename):
    """Print a table to console and save it as a CSV.

    Args:
        df: DataFrame to display and save.
        title: Section title string.
        filename: Output CSV filename (within RESULTS_DIR).
    """
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)
    if df is not None and not df.empty:
        print(df.to_string(index=False))
        df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
        print(f"  Saved: results/tables/{filename}")
    else:
        print("  [N/A] Data not available — run experiment pipelines first.")


# ─────────────────────── Main ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SMARTSTOCK — Generate All Paper Results")
    print("=" * 60)

    total_start = time.perf_counter()

    # Run forecasting pipeline
    ok = run_pipeline("src/forecasting/run_all_forecasting.py",
                      "Running Forecasting Experiment Pipeline")
    if not ok:
        print("[WARN] Forecasting pipeline failed — will compile whatever results exist.")

    # Run NLP pipeline
    ok = run_pipeline("src/nlp/run_all_nlp.py",
                      "Running NLP Experiment Pipeline")
    if not ok:
        print("[WARN] NLP pipeline failed — will compile whatever results exist.")

    # Run integration demo
    ok = run_pipeline("src/integration/pipeline.py",
                      "Running Integration Pipeline Demo")
    if not ok:
        print("[WARN] Integration pipeline failed — pipeline summary may be missing.")

    # ── Compile paper tables ──
    print("\n" + "=" * 60)
    print("COMPILING PAPER TABLES")
    print("=" * 60)

    print_and_save_table(build_table1(), "Table 1: Forecasting Model Comparison",      "paper_table1_forecasting.csv")
    print_and_save_table(build_table2(), "Table 2: Forecasting Compression for Edge",  "paper_table2_compression.csv")
    print_and_save_table(build_table3(), "Table 3: NLP Intent Classification",         "paper_table3_nlp.csv")
    print_and_save_table(build_table4(), "Table 4: NLP Compression for Edge",          "paper_table4_nlp_compression.csv")
    print_and_save_table(build_table5(), "Table 5: End-to-End Pipeline Performance",   "paper_table5_pipeline.csv")

    total = time.perf_counter() - total_start
    print(f"\n{'=' * 60}")
    print(f"ALL DONE — Total time: {total:.1f}s")
    print(f"Tables : results/tables/paper_table*.csv")
    print(f"Plots  : results/plots/*.png")
    print(f"Models : results/models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
