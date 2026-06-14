"""
Master runner for the full forecasting experiment pipeline.

Executes all forecasting steps in the correct order. Each step prints a
separator so progress is easy to follow.

Usage:
    python src/forecasting/run_all_forecasting.py
"""

import subprocess
import sys
import time


STEPS = [
    ("src/forecasting/data_preprocessing.py", "STEP 1: Data Preprocessing"),
    ("src/forecasting/model_moving_avg.py",    "STEP 2: Moving Average Baseline"),
    ("src/forecasting/model_prophet.py",       "STEP 3: Prophet Model"),
    ("src/forecasting/model_lstm.py",          "STEP 4: LSTM Model"),
    ("src/forecasting/model_cnn.py",           "STEP 5: CNN Model"),
    ("src/forecasting/model_compression.py",   "STEP 6: Model Compression"),
    ("src/forecasting/evaluate.py",            "STEP 7: Evaluation & Plots"),
]


def run_step(script_path, step_label):
    """Run a single pipeline step as a subprocess.

    Args:
        script_path: Relative path to the Python script.
        step_label: Human-readable label printed in the separator.

    Returns:
        True if the step succeeded, False otherwise.
    """
    print("\n" + "=" * 60)
    print(step_label)
    print("=" * 60)
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, script_path])
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {script_path} failed with return code {result.returncode}")
        return False
    print(f"\n  Completed in {elapsed:.1f}s")
    return True


def main():
    print("=" * 60)
    print("SMARTSTOCK — Forecasting Experiment Pipeline")
    print("=" * 60)

    total_start = time.perf_counter()
    for script, label in STEPS:
        ok = run_step(script, label)
        if not ok:
            print("\n[ABORT] Pipeline stopped due to error. Fix the issue and re-run.")
            sys.exit(1)

    total = time.perf_counter() - total_start
    print("\n" + "=" * 60)
    print(f"ALL FORECASTING STEPS COMPLETE — Total time: {total:.1f}s")
    print("Results saved in results/tables/ and results/plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
