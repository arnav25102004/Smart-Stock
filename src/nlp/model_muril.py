"""
Fine-tuned MuRIL (Multilingual Representations for Indian Languages) model.

Downloads 'google/muril-base-cased' from HuggingFace and fine-tunes it for
5-class intent classification using the SMARTSTOCK NLP dataset.

CPU tip: If training on a CPU-only machine, the script automatically reduces
epochs to 5 and batch_size to 8. Use Google Colab (free GPU) for full config.

Usage:
    python src/nlp/model_muril.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.metrics import (
    all_classification_metrics,
    get_classification_report,
    get_confusion_matrix,
)

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from datasets import Dataset
except ImportError:
    print("[ERROR] transformers / torch / datasets not installed.")
    print("  Run: pip install transformers torch datasets")
    sys.exit(1)

NLP_DIR = "data/nlp_dataset"
RESULTS_DIR = "results/tables"
MODELS_DIR = "results/models/muril_finetuned"
RANDOM_SEED = 42
MODEL_NAME = "google/muril-base-cased"
MAX_LENGTH = 64

INTENTS = ["check_stock", "add_stock", "get_prediction", "low_stock_alert", "help"]
LABEL2ID = {intent: i for i, intent in enumerate(INTENTS)}
ID2LABEL = {i: intent for i, intent in enumerate(INTENTS)}

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}")

# Auto-reduce config for CPU-only machines
if DEVICE == "cpu":
    EPOCHS = 5
    BATCH_SIZE = 8
    print("  [INFO] CPU detected — using reduced config (5 epochs, batch=8).")
    print("  For full training, run on Google Colab with GPU.")
else:
    EPOCHS = 10
    BATCH_SIZE = 16


def load_splits(df):
    """Load the canonical train/val/test split shared across all NLP models.

    Args:
        df: Full dataset DataFrame.

    Returns:
        Tuple (train_df, val_df, test_df).
    """
    split_path = os.path.join(NLP_DIR, "nlp_split_indices.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            meta = json.load(f)
        return (df.loc[meta["train_indices"]],
                df.loc[meta["val_indices"]],
                df.loc[meta["test_indices"]])
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED,
                                         stratify=df["intent"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED,
                                       stratify=temp_df["intent"])
    return train_df, val_df, test_df


def tokenize_dataset(tokenizer, df):
    """Tokenize a DataFrame into a HuggingFace Dataset.

    Args:
        tokenizer: HuggingFace tokenizer for MuRIL.
        df: DataFrame with 'sentence' and 'intent' columns.

    Returns:
        HuggingFace Dataset with input_ids, attention_mask, labels.
    """
    records = {
        "text": df["sentence"].tolist(),
        "label": [LABEL2ID[i] for i in df["intent"].tolist()]
    }
    hf_ds = Dataset.from_dict(records)

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True
        )

    hf_ds = hf_ds.map(_tokenize, batched=True)
    hf_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return hf_ds


def compute_metrics(eval_pred):
    """Compute accuracy for HuggingFace Trainer.

    Args:
        eval_pred: EvalPrediction object with logits and label_ids.

    Returns:
        Dict with 'accuracy' key.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def measure_inference_time(model, tokenizer, sentences, n_runs=3):
    """Time MuRIL inference for a list of sentences.

    Args:
        model: Fine-tuned PyTorch model (in eval mode).
        tokenizer: HuggingFace tokenizer.
        sentences: List of sentence strings.
        n_runs: Number of full-pass repetitions for stable timing.

    Returns:
        Average per-sentence inference time in milliseconds.
    """
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            for sent in sentences:
                enc = tokenizer(sent, return_tensors="pt", max_length=MAX_LENGTH,
                                padding="max_length", truncation=True)
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                _ = model(**enc)
            times.append(time.perf_counter() - t0)
    return (np.mean(times) / len(sentences)) * 1000


def predict_all(model, tokenizer, sentences):
    """Run batch inference and return predicted label indices.

    Args:
        model: Fine-tuned PyTorch model.
        tokenizer: HuggingFace tokenizer.
        sentences: List of sentence strings.

    Returns:
        numpy array of predicted intent label indices.
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for sent in sentences:
            enc = tokenizer(sent, return_tensors="pt", max_length=MAX_LENGTH,
                            padding="max_length", truncation=True)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = model(**enc).logits
            all_preds.append(torch.argmax(logits, dim=-1).item())
    return np.array(all_preds)


def main():
    print("=" * 60)
    print("STEP 11: Fine-tuning MuRIL")
    print("=" * 60)

    dataset_path = os.path.join(NLP_DIR, "smartstock_nlp_dataset.csv")
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print("  Run dataset_creator.py first.")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    train_df, val_df, test_df = load_splits(df)
    print(f"\n  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    print(f"\n  Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(INTENTS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    print("\n  Tokenizing datasets...")
    train_ds = tokenize_dataset(tokenizer, train_df)
    val_ds = tokenize_dataset(tokenizer, val_df)
    test_ds = tokenize_dataset(tokenizer, test_df)

    training_args = TrainingArguments(
        output_dir=MODELS_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        seed=RANDOM_SEED,
        no_cuda=(DEVICE == "cpu"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"\n  Training MuRIL ({EPOCHS} epochs, batch={BATCH_SIZE})...")
    t0 = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Save fine-tuned model
    trainer.save_model(MODELS_DIR)
    tokenizer.save_pretrained(MODELS_DIR)

    # Measure saved model size
    total_size = sum(
        os.path.getsize(os.path.join(MODELS_DIR, f))
        for f in os.listdir(MODELS_DIR)
        if os.path.isfile(os.path.join(MODELS_DIR, f))
    )
    model_size_mb = total_size / (1024 * 1024)
    print(f"  Model size: {model_size_mb:.2f} MB")

    # Evaluate on test set
    print("\n  Evaluating on test set...")
    y_pred_ids = predict_all(model, tokenizer, test_df["sentence"].tolist())
    y_pred = np.array([ID2LABEL[i] for i in y_pred_ids])
    y_true = test_df["intent"].values

    metrics = all_classification_metrics(y_true, y_pred)
    inf_ms = measure_inference_time(model, tokenizer, test_df["sentence"].tolist()[:50])

    print(f"\n  Accuracy  : {metrics['Accuracy (%)']:.2f}%")
    print(f"  Macro F1  : {metrics['F1 (macro)']:.4f}")
    print(f"  Inference : {inf_ms:.4f} ms/sentence")
    print(f"  Model size: {model_size_mb:.2f} MB ({model_size_mb * 1024:.0f} KB)")

    print("\n  === Per-Language Accuracy ===")
    lang_accs = {}
    for lang in test_df["language"].unique():
        sub = test_df[test_df["language"] == lang]
        sub_preds = predict_all(model, tokenizer, sub["sentence"].tolist())
        sub_pred_labels = np.array([ID2LABEL[i] for i in sub_preds])
        acc = (sub_pred_labels == sub["intent"].values).mean() * 100
        lang_accs[lang] = round(acc, 2)
        print(f"  {lang:15s}: {acc:.2f}%")

    print("\n  Classification Report:")
    print(get_classification_report(y_true, y_pred, labels=INTENTS))

    cm = get_confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=INTENTS, columns=INTENTS).to_csv(
        f"{RESULTS_DIR}/nlp_muril_confusion.csv"
    )

    result_row = {
        "Model": "MuRIL (fine-tuned)",
        **metrics,
        "Training_Time_s": round(train_time, 2),
        "Inference_Time_ms": round(inf_ms, 4),
        "Model_Size_KB": round(model_size_mb * 1024, 1),
        "Num_Parameters": n_params,
        **{f"{k}_Acc": v for k, v in lang_accs.items()}
    }
    pd.DataFrame([result_row]).to_csv(f"{RESULTS_DIR}/nlp_muril.csv", index=False)

    detail_df = test_df.copy()
    detail_df["predicted"] = y_pred
    detail_df.to_csv(f"{RESULTS_DIR}/nlp_muril_predictions.csv", index=False)

    print(f"\n  Saved fine-tuned model to: {MODELS_DIR}/")
    print(f"  Saved: {RESULTS_DIR}/nlp_muril.csv")
    print("  [DONE] MuRIL fine-tuning complete.")


if __name__ == "__main__":
    main()
