"""
TF-IDF + SVM intent classifier for multilingual inventory queries.

Uses character n-gram TF-IDF (works better than word n-grams for Romanized
Indian languages) with a linear SVM. Also trains a Random Forest and reports
whichever performs better on validation accuracy.

Usage:
    python src/nlp/model_tfidf_svm.py
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.metrics import (
    all_classification_metrics,
    get_classification_report,
    get_confusion_matrix,
)

NLP_DIR = "data/nlp_dataset"
RESULTS_DIR = "results/tables"
MODELS_DIR = "results/models"
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

INTENTS = ["check_stock", "add_stock", "get_prediction", "low_stock_alert", "help"]


def load_splits(df):
    """Load the same train/val/test split used by all NLP models.

    Falls back to a fresh stratified split if the indices file is missing.

    Args:
        df: Full dataset DataFrame.

    Returns:
        Tuple (train_df, val_df, test_df).
    """
    split_path = os.path.join(NLP_DIR, "nlp_split_indices.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            meta = json.load(f)
        train_df = df.loc[meta["train_indices"]]
        val_df = df.loc[meta["val_indices"]]
        test_df = df.loc[meta["test_indices"]]
    else:
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED,
                                             stratify=df["intent"])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED,
                                           stratify=temp_df["intent"])
    return train_df, val_df, test_df


def build_tfidf_svm():
    """Return a sklearn Pipeline with char-ngram TF-IDF and linear SVM.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=10_000,
            sublinear_tf=True
        )),
        ("clf", LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_SEED))
    ])


def build_tfidf_rf():
    """Return a sklearn Pipeline with char-ngram TF-IDF and Random Forest.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=10_000,
            sublinear_tf=True
        )),
        ("clf", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
        ))
    ])


def measure_inference_time(pipeline, sentences, n_runs=3):
    """Time inference for a fitted pipeline, averaged over multiple passes.

    Args:
        pipeline: Fitted sklearn Pipeline.
        sentences: List/array of sentence strings.
        n_runs: Number of passes to average over.

    Returns:
        Average per-sentence inference time in milliseconds.
    """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        pipeline.predict(sentences)
        times.append(time.perf_counter() - t0)
    return (np.mean(times) / len(sentences)) * 1000


def save_model_and_measure_size(pipeline, name):
    """Pickle the pipeline and return its file size in KB.

    Args:
        pipeline: Fitted sklearn Pipeline.
        name: File stem (without extension).

    Returns:
        File size in KB.
    """
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    return os.path.getsize(path) / 1024


def evaluate_per_language(pipeline, test_df):
    """Compute accuracy broken down by language.

    Args:
        pipeline: Fitted sklearn Pipeline.
        test_df: Test DataFrame with 'sentence', 'intent', 'language' columns.

    Returns:
        Dict mapping language → accuracy (%).
    """
    lang_accs = {}
    for lang in test_df["language"].unique():
        sub = test_df[test_df["language"] == lang]
        preds = pipeline.predict(sub["sentence"].values)
        acc = (preds == sub["intent"].values).mean() * 100
        lang_accs[lang] = round(acc, 2)
    return lang_accs


def main():
    print("=" * 60)
    print("STEP 10: TF-IDF + SVM/RF Classifier")
    print("=" * 60)

    dataset_path = os.path.join(NLP_DIR, "smartstock_nlp_dataset.csv")
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print("  Run dataset_creator.py first.")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    train_df, val_df, test_df = load_splits(df)
    print(f"\n  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    X_train = train_df["sentence"].values
    y_train = train_df["intent"].values
    X_val = val_df["sentence"].values
    y_val = val_df["intent"].values
    X_test = test_df["sentence"].values
    y_test = test_df["intent"].values

    # ── Train SVM ──
    print("\n  Training TF-IDF + LinearSVM...")
    t0 = time.perf_counter()
    svm_pipe = build_tfidf_svm()
    svm_pipe.fit(X_train, y_train)
    svm_train_time = time.perf_counter() - t0
    svm_val_acc = (svm_pipe.predict(X_val) == y_val).mean() * 100
    print(f"  SVM val accuracy: {svm_val_acc:.2f}%  Train time: {svm_train_time:.2f}s")

    # ── Train Random Forest ──
    print("\n  Training TF-IDF + RandomForest...")
    t0 = time.perf_counter()
    rf_pipe = build_tfidf_rf()
    rf_pipe.fit(X_train, y_train)
    rf_train_time = time.perf_counter() - t0
    rf_val_acc = (rf_pipe.predict(X_val) == y_val).mean() * 100
    print(f"  RF  val accuracy: {rf_val_acc:.2f}%  Train time: {rf_train_time:.2f}s")

    # ── Pick best ──
    if svm_val_acc >= rf_val_acc:
        best_pipe = svm_pipe
        best_name = "TF-IDF + SVM"
        best_train_time = svm_train_time
        print(f"\n  Selected: {best_name}")
    else:
        best_pipe = rf_pipe
        best_name = "TF-IDF + RF"
        best_train_time = rf_train_time
        print(f"\n  Selected: {best_name}")

    # ── Evaluate on test ──
    y_pred = best_pipe.predict(X_test)
    metrics = all_classification_metrics(y_test, y_pred)
    inf_ms = measure_inference_time(best_pipe, X_test)
    size_kb = save_model_and_measure_size(best_pipe, "tfidf_svm")

    print(f"\n  Accuracy  : {metrics['Accuracy (%)']:.2f}%")
    print(f"  Macro F1  : {metrics['F1 (macro)']:.4f}")
    print(f"  Inference : {inf_ms:.4f} ms/sentence")
    print(f"  Model size: {size_kb:.2f} KB")

    print("\n  === Per-Language Accuracy ===")
    lang_accs = evaluate_per_language(best_pipe, test_df)
    for lang, acc in lang_accs.items():
        print(f"  {lang:15s}: {acc:.2f}%")

    print("\n  Classification Report:")
    print(get_classification_report(y_test, y_pred, labels=INTENTS))

    # ── Confusion matrix ──
    cm = get_confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=INTENTS, columns=INTENTS)
    cm_df.to_csv(f"{RESULTS_DIR}/nlp_tfidf_svm_confusion.csv")

    # ── Save results ──
    result_row = {
        "Model": best_name,
        **metrics,
        "Training_Time_s": round(best_train_time, 2),
        "Inference_Time_ms": round(inf_ms, 4),
        "Model_Size_KB": round(size_kb, 2),
        **{f"{k}_Acc": v for k, v in lang_accs.items()}
    }
    pd.DataFrame([result_row]).to_csv(f"{RESULTS_DIR}/nlp_tfidf_svm.csv", index=False)

    detail_df = test_df.copy()
    detail_df["predicted"] = y_pred
    detail_df.to_csv(f"{RESULTS_DIR}/nlp_tfidf_svm_predictions.csv", index=False)

    print(f"\n  Saved: {RESULTS_DIR}/nlp_tfidf_svm.csv")
    print("  [DONE] TF-IDF + SVM/RF classifier complete.")


if __name__ == "__main__":
    main()
