"""
Master evaluation script for the NLP experiment.

Loads per-model result CSVs, compiles the master comparison table,
and generates all publication-ready plots.

Usage:
    python src/nlp/evaluate_nlp.py
    (Run after all model_*.py NLP scripts have completed.)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import visualization as viz
from src.utils.metrics import get_confusion_matrix

RESULTS_DIR = "results/tables"
PLOTS_DIR = "results/plots"
NLP_DIR = "data/nlp_dataset"

INTENTS = ["check_stock", "add_stock", "get_prediction", "low_stock_alert", "help"]
LANGUAGES = ["English", "Hindi", "Kannada", "Code-Mixed"]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_result(filename):
    """Load a model result CSV, returning None if missing.

    Args:
        filename: Filename within RESULTS_DIR.

    Returns:
        Single-row DataFrame or None.
    """
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"  [WARN] Missing: {path}")
    return None


def safe_lang_acc(row, lang):
    """Safely extract per-language accuracy from a result row.

    Args:
        row: pandas Series (one model's results).
        lang: Language name string.

    Returns:
        Float accuracy or 0.0 if not present.
    """
    col = f"{lang}_Acc"
    if col in row.index:
        return float(row[col])
    return 0.0


def build_master_table(summaries):
    """Build the NLP comparison table from per-model summary DataFrames.

    Args:
        summaries: List of single-row DataFrames (one per model).

    Returns:
        Combined DataFrame.
    """
    rows = []
    for df in summaries:
        if df is None:
            continue
        row = df.iloc[0]
        rows.append({
            "Model": row.get("Model", "Unknown"),
            "English Acc (%)": safe_lang_acc(row, "English"),
            "Hindi Acc (%)": safe_lang_acc(row, "Hindi"),
            "Kannada Acc (%)": safe_lang_acc(row, "Kannada"),
            "Code-Mixed Acc (%)": safe_lang_acc(row, "Code-Mixed"),
            "Overall Acc (%)": row.get("Accuracy (%)", 0.0),
            "F1 (macro)": row.get("F1 (macro)", 0.0),
            "Model_Size_KB": row.get("Model_Size_KB", 0.0),
            "Inference_Time_ms": row.get("Inference_Time_ms", 0.0),
        })
    return pd.DataFrame(rows)


def plot_confusion_matrices():
    """Plot confusion matrix heatmaps for MuRIL (best) and Rule-Based (worst)."""
    for model_key, label in [("muril", "MuRIL"), ("rule_based", "Rule-Based")]:
        cm_path = os.path.join(RESULTS_DIR, f"nlp_{model_key}_confusion.csv")
        if not os.path.exists(cm_path):
            continue
        cm_df = pd.read_csv(cm_path, index_col=0)
        viz.plot_confusion_matrix(
            cm_df.values,
            list(cm_df.columns),
            label,
            f"nlp_confusion_{model_key}"
        )
    print("  Saved confusion matrix plots.")


def plot_per_intent_f1_chart(master):
    """Generate per-intent F1 grouped bar chart if prediction CSVs exist.

    Args:
        master: Master NLP comparison DataFrame.
    """
    model_files = {
        "Rule-Based": "nlp_rule_based_predictions.csv",
        "TF-IDF+SVM": "nlp_tfidf_svm_predictions.csv",
        "MuRIL": "nlp_muril_predictions.csv",
    }

    intent_f1s = {intent: [] for intent in INTENTS}
    model_names = []

    for model_label, fname in model_files.items():
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        model_names.append(model_label)
        for intent in INTENTS:
            tp = ((df["intent"] == intent) & (df["predicted"] == intent)).sum()
            fp = ((df["intent"] != intent) & (df["predicted"] == intent)).sum()
            fn = ((df["intent"] == intent) & (df["predicted"] != intent)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            intent_f1s[intent].append(round(f1, 4))

    if model_names:
        viz.plot_per_intent_f1(model_names, intent_f1s, INTENTS, "nlp_per_intent_f1")
        print("  Saved per-intent F1 chart.")


def main():
    print("=" * 60)
    print("STEP 13: Master NLP Evaluation")
    print("=" * 60)

    print("\n  [1/5] Building master comparison table...")
    summaries = [
        load_result("nlp_rule_based.csv"),
        load_result("nlp_tfidf_svm.csv"),
        load_result("nlp_muril.csv"),
    ]
    master = build_master_table(summaries)

    if master.empty:
        print("[ERROR] No NLP model results found. Run all model scripts first.")
        sys.exit(1)

    master_path = os.path.join(RESULTS_DIR, "nlp_comparison.csv")
    master.to_csv(master_path, index=False)
    print(f"\n{master.to_string(index=False)}")

    model_names = master["Model"].tolist()
    overall_accs = master["Overall Acc (%)"].tolist()

    print("\n  [2/5] Plotting per-language accuracy...")
    lang_accs = {
        "English": master["English Acc (%)"].tolist(),
        "Hindi": master["Hindi Acc (%)"].tolist(),
        "Kannada": master["Kannada Acc (%)"].tolist(),
        "Code-Mixed": master["Code-Mixed Acc (%)"].tolist(),
    }
    viz.plot_per_language_accuracy(model_names, lang_accs, "nlp_per_language_accuracy")
    print("  Saved per-language accuracy chart.")

    print("\n  [3/5] Plotting confusion matrices...")
    plot_confusion_matrices()

    print("\n  [4/5] Plotting NLP compression comparison...")
    comp_path = os.path.join(RESULTS_DIR, "nlp_compression_comparison.csv")
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        versions = comp_df["Version"].tolist()
        sizes = comp_df["Size_KB"].tolist()
        full_sizes = [comp_df[comp_df["Version"] == "Full PyTorch"]["Size_KB"].values[0]] * len(versions)
        viz.plot_compression_comparison(versions, full_sizes, sizes, "nlp_compression_comparison")
        print("  Saved NLP compression chart.")
    else:
        print("  [SKIP] nlp_compression_comparison.csv not found.")

    print("\n  [5/5] Plotting per-intent F1...")
    plot_per_intent_f1_chart(master)

    print(f"\n  Saved: {master_path}")
    print(f"  All plots saved to: {PLOTS_DIR}/")
    print("  [DONE] NLP evaluation complete.")


if __name__ == "__main__":
    main()
