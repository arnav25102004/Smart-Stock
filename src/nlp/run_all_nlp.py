"""
Master runner for the full NLP experiment pipeline.

Executes all NLP steps in the correct order with clear progress separators.

Usage:
    python src/nlp/run_all_nlp.py
"""

import subprocess
import sys
import time

STEPS = [
    ("src/nlp/dataset_creator.py",       "STEP 8:  NLP Dataset Creation"),
    ("src/nlp/model_rule_based.py",       "STEP 9:  Rule-Based Classifier"),
    ("src/nlp/model_tfidf_svm.py",        "STEP 10: TF-IDF + SVM Classifier"),
    ("src/nlp/model_muril.py",            "STEP 11: MuRIL Fine-Tuning"),
    ("src/nlp/model_muril_compressed.py", "STEP 12: MuRIL Compression"),
    ("src/nlp/evaluate_nlp.py",           "STEP 13: NLP Evaluation & Plots"),
]


def run_step(script_path, step_label):
    """Run a single pipeline step as a subprocess.

    Args:
        script_path: Relative path to the Python script.
        step_label: Human-readable step label for console output.

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
        print(f"\n[ERROR] {script_path} failed (return code {result.returncode})")
        return False
    print(f"\n  Completed in {elapsed:.1f}s")
    return True


def main():
    print("=" * 60)
    print("SMARTSTOCK — NLP Experiment Pipeline")
    print("=" * 60)

    total_start = time.perf_counter()
    for script, label in STEPS:
        ok = run_step(script, label)
        if not ok:
            print("\n[ABORT] Pipeline stopped due to error.")
            sys.exit(1)

    total = time.perf_counter() - total_start
    print("\n" + "=" * 60)
    print(f"ALL NLP STEPS COMPLETE — Total time: {total:.1f}s")
    print("Results saved in results/tables/ and results/plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
