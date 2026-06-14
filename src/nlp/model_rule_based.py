"""
Rule-based keyword matching baseline for multilingual intent classification.

Classifies each sentence by counting keyword matches across intent-specific
keyword lists. If no keywords match, defaults to 'help'.
Provides per-language accuracy breakdown for the paper.

Usage:
    python src/nlp/model_rule_based.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.metrics import all_classification_metrics, get_classification_report, get_confusion_matrix

NLP_DIR = "data/nlp_dataset"
RESULTS_DIR = "results/tables"
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

INTENTS = ["check_stock", "add_stock", "get_prediction", "low_stock_alert", "help"]

# Keyword lists cover English, Hindi (Romanized), and Kannada (Romanized) terms
KEYWORDS = {
    "check_stock": [
        "kitna", "eshtu", "how much", "stock", "bacha", "left", "remaining",
        "check", "ide", "hai kya", "available", "inventory", "level", "uliyitu",
        "tilikoli", "dikhao", "batao", "count", "show", "current", "kya hai",
        "kitni", "kitnee", "konchaa", "status", "summary", "report",
    ],
    "add_stock": [
        "add", "daal", "restock", "haaku", "increase", "put", "refill",
        "order", "bharti", "mangao", "kharido", "daalo", "jaasti", "taa",
        "bring", "purchase", "get more", "replenish", "top up", "arrived",
        "delivery", "received", "new stock", "hosa", "naya", "badhao",
    ],
    "get_prediction": [
        "kal", "tomorrow", "prediction", "forecast", "naale", "lagega",
        "chahiye", "estimate", "bikega", "demand", "expect", "agle",
        "munde", "next week", "next month", "will sell", "will need",
        "andaza", "hisab", "plan", "predict", "bikuna", "bedu",
    ],
    "low_stock_alert": [
        "khatam", "low", "alert", "kam", "finish", "mugidu", "running out",
        "warning", "shortage", "empty", "critical", "urgent", "zero",
        "almost", "nearly", "mugiyutte", "toot", "nahi bacha", "illa",
        "threshold", "minimum", "below", "out of stock", "finished",
    ],
    "help": [
        "help", "kaise", "how to", "use", "guide", "sahayata", "manual",
        "tutorial", "hege", "explain", "features", "start", "instructions",
        "what can", "what is", "getting started", "usage", "heli maadi",
        "samjhao", "batao kaise", "sikhaao",
    ],
}


def classify(sentence):
    """Classify a sentence by counting keyword matches per intent.

    Args:
        sentence: Raw input string (any language, any script).

    Returns:
        Predicted intent label string.
    """
    s = sentence.lower()
    scores = {intent: 0 for intent in INTENTS}
    for intent, kws in KEYWORDS.items():
        for kw in kws:
            if kw in s:
                scores[intent] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "help"


def measure_inference_time(sentences, n_runs=3):
    """Time classification over the full sentence list, averaged over n_runs.

    Args:
        sentences: List/Series of sentence strings.
        n_runs: Number of full-pass repetitions for stable timing.

    Returns:
        Average per-sentence inference time in milliseconds.
    """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for s in sentences:
            classify(s)
        times.append(time.perf_counter() - t0)
    total_avg = np.mean(times)
    return (total_avg / len(sentences)) * 1000


def main():
    print("=" * 60)
    print("STEP 9: Rule-Based Keyword Classifier")
    print("=" * 60)

    dataset_path = os.path.join(NLP_DIR, "smartstock_nlp_dataset.csv")
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print("  Run dataset_creator.py first.")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    print(f"\n  Loaded {len(df)} sentences.")

    # Stratified split: 80/10/10 by intent
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED,
                                         stratify=df["intent"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED,
                                       stratify=temp_df["intent"])
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # Save split indices for consistency across models
    split_path = os.path.join(NLP_DIR, "nlp_split_indices.json")
    if not os.path.exists(split_path):
        split_meta = {
            "train_indices": train_df.index.tolist(),
            "val_indices": val_df.index.tolist(),
            "test_indices": test_df.index.tolist(),
        }
        with open(split_path, "w") as f:
            json.dump(split_meta, f)
        print(f"  Saved split indices → {split_path}")

    print("\n  Classifying test set...")
    y_true = test_df["intent"].values
    y_pred = test_df["sentence"].apply(classify).values

    metrics = all_classification_metrics(y_true, y_pred)
    inf_ms = measure_inference_time(test_df["sentence"].tolist())

    print(f"\n  Overall Accuracy : {metrics['Accuracy (%)']:.2f}%")
    print(f"  Macro F1         : {metrics['F1 (macro)']:.4f}")
    print(f"  Inference time   : {inf_ms:.4f} ms/sentence")

    print("\n  === Per-Language Accuracy ===")
    lang_rows = []
    for lang in test_df["language"].unique():
        sub = test_df[test_df["language"] == lang]
        sub_true = sub["intent"].values
        sub_pred = sub["sentence"].apply(classify).values
        acc = (sub_true == sub_pred).mean() * 100
        print(f"  {lang:15s}: {acc:.2f}%")
        lang_rows.append({"Language": lang, "Accuracy (%)": round(acc, 2)})

    print("\n  Classification Report:")
    print(get_classification_report(y_true, y_pred, labels=INTENTS))

    # Save outputs
    results = pd.DataFrame([{
        "Model": "Rule-Based",
        **metrics,
        "Inference_Time_ms": round(inf_ms, 4),
        "Model_Size_KB": 0.0,
        **{f"{r['Language']}_Acc": r["Accuracy (%)"] for r in lang_rows}
    }])
    results.to_csv(f"{RESULTS_DIR}/nlp_rule_based.csv", index=False)

    detail_df = test_df.copy()
    detail_df["predicted"] = y_pred
    detail_df.to_csv(f"{RESULTS_DIR}/nlp_rule_based_predictions.csv", index=False)

    pd.DataFrame(lang_rows).to_csv(f"{RESULTS_DIR}/nlp_rule_based_by_language.csv", index=False)

    print(f"\n  Saved: {RESULTS_DIR}/nlp_rule_based.csv")
    print("  [DONE] Rule-based classifier complete.")


if __name__ == "__main__":
    main()
