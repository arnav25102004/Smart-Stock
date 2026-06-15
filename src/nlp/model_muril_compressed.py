"""
Edge-deployment compression of the fine-tuned MuRIL model.

Creates three versions:
  1. Full PyTorch model (baseline)
  2. ONNX export
  3. Dynamically quantized ONNX (QInt8 weights)

Compares accuracy, model size, and inference speed across all three versions
on the same test set.

Usage:
    python src/nlp/model_muril_compressed.py
    (Requires model_muril.py to have been run first.)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.metrics import all_classification_metrics

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    print("[ERROR] transformers / torch not installed.")
    sys.exit(1)

NLP_DIR = "data/nlp_dataset"
RESULTS_DIR = "results/tables"
MODELS_DIR = "results/models/muril_finetuned"
ONNX_DIR = "results/models/muril_onnx"
RANDOM_SEED = 42
MAX_LENGTH = 64

INTENTS = ["check_stock", "add_stock", "get_prediction", "low_stock_alert", "help"]
LABEL2ID = {intent: i for i, intent in enumerate(INTENTS)}
ID2LABEL = {i: intent for i, intent in enumerate(INTENTS)}

os.makedirs(ONNX_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_test_data(df):
    """Load the canonical test split.

    Args:
        df: Full dataset DataFrame.

    Returns:
        Test DataFrame slice.
    """
    split_path = os.path.join(NLP_DIR, "nlp_split_indices.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            meta = json.load(f)
        return df.loc[meta["test_indices"]]
    from sklearn.model_selection import train_test_split
    _, temp = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["intent"])
    _, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_SEED, stratify=temp["intent"])
    return test


def pytorch_predict_all(model, tokenizer, sentences):
    """Run full PyTorch inference on a list of sentences.

    Args:
        model: PyTorch model in eval mode.
        tokenizer: HuggingFace tokenizer.
        sentences: List of sentence strings.

    Returns:
        Tuple (predicted_labels list, avg_inference_ms).
    """
    model.eval()
    preds, times = [], []
    with torch.no_grad():
        for sent in sentences:
            enc = tokenizer(sent, return_tensors="pt", max_length=MAX_LENGTH,
                            padding="max_length", truncation=True)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            t0 = time.perf_counter()
            logits = model(**enc).logits
            times.append(time.perf_counter() - t0)
            preds.append(ID2LABEL[torch.argmax(logits, dim=-1).item()])
    return preds, np.mean(times) * 1000


def export_to_onnx(model, tokenizer, onnx_path):
    """Export the PyTorch model to ONNX format.

    Args:
        model: Fine-tuned PyTorch model.
        tokenizer: HuggingFace tokenizer.
        onnx_path: Output path for the .onnx file.
    """
    model.eval()
    dummy = tokenizer("dummy sentence", return_tensors="pt",
                      max_length=MAX_LENGTH, padding="max_length", truncation=True)
    dummy = {k: v.to(DEVICE) for k, v in dummy.items()}

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"  ONNX model saved: {onnx_path}")


def onnx_predict_all(onnx_path, tokenizer, sentences):
    """Run inference using the ONNX Runtime.

    Args:
        onnx_path: Path to the .onnx model file.
        tokenizer: HuggingFace tokenizer.
        sentences: List of sentence strings.

    Returns:
        Tuple (predicted_labels list, avg_inference_ms).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime not installed. Run: pip install onnxruntime")
        return [], 0.0

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    preds, times = [], []
    for sent in sentences:
        enc = tokenizer(sent, return_tensors="np", max_length=MAX_LENGTH,
                        padding="max_length", truncation=True)
        t0 = time.perf_counter()
        logits = sess.run(["logits"], {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64)
        })[0]
        times.append(time.perf_counter() - t0)
        preds.append(ID2LABEL[int(np.argmax(logits, axis=-1)[0])])
    return preds, np.mean(times) * 1000


def quantize_onnx(onnx_path, quant_path):
    """Apply dynamic INT8 quantization to an ONNX model.

    Args:
        onnx_path: Path to the full ONNX model.
        quant_path: Output path for the quantized ONNX model.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("[ERROR] onnxruntime.quantization not available.")
        return False
    quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
    print(f"  Quantized ONNX saved: {quant_path}")
    return True


def folder_size_kb(folder):
    """Sum file sizes in a directory and return the total in KB.

    Args:
        folder: Directory path.

    Returns:
        Total size in KB.
    """
    total = sum(
        os.path.getsize(os.path.join(folder, f))
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    )
    return total / 1024


def per_language_accuracy(preds, test_df):
    """Compute per-language accuracy.

    Args:
        preds: List of predicted intent label strings.
        test_df: DataFrame with 'language' and 'intent' columns.

    Returns:
        Dict mapping language → accuracy (%).
    """
    test_df = test_df.copy()
    test_df["predicted"] = preds
    lang_accs = {}
    for lang in test_df["language"].unique():
        sub = test_df[test_df["language"] == lang]
        lang_accs[lang] = round((sub["predicted"] == sub["intent"]).mean() * 100, 2)
    return lang_accs


def main():
    print("=" * 60)
    print("STEP 12: MuRIL Compression for Edge")
    print("=" * 60)

    if not os.path.exists(MODELS_DIR):
        print(f"[ERROR] Fine-tuned model not found at: {MODELS_DIR}")
        print("  Run model_muril.py first.")
        sys.exit(1)

    dataset_path = os.path.join(NLP_DIR, "smartstock_nlp_dataset.csv")
    df = pd.read_csv(dataset_path)
    test_df = load_test_data(df)
    sentences = test_df["sentence"].tolist()
    y_true = test_df["intent"].values
    print(f"\n  Test samples: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR)
    model.to(DEVICE)

    onnx_path = os.path.join(ONNX_DIR, "muril.onnx")
    quant_path = os.path.join(ONNX_DIR, "muril_quantized.onnx")

    all_results = []

    # ── Version 1: Full PyTorch ──
    print("\n  [1/3] Full PyTorch model...")
    preds_pt, inf_ms_pt = pytorch_predict_all(model, tokenizer, sentences)
    m = all_classification_metrics(y_true, np.array(preds_pt))
    size_kb = folder_size_kb(MODELS_DIR)
    lang_accs = per_language_accuracy(preds_pt, test_df)
    print(f"  Accuracy: {m['Accuracy (%)']:.2f}%  Size: {size_kb:.0f} KB  Inference: {inf_ms_pt:.4f} ms")
    all_results.append({"Version": "Full PyTorch", "Size_KB": round(size_kb, 1),
                         **m, "Inference_Time_ms": round(inf_ms_pt, 4),
                         "Compression_Ratio": 1.0, **{f"{k}_Acc": v for k, v in lang_accs.items()}})
    full_size_kb = size_kb

    # ── Version 2: ONNX ──
    print("\n  [2/3] ONNX export...")
    export_to_onnx(model, tokenizer, onnx_path)
    onnx_size_kb = os.path.getsize(onnx_path) / 1024
    preds_onnx, inf_ms_onnx = onnx_predict_all(onnx_path, tokenizer, sentences)
    if preds_onnx:
        m = all_classification_metrics(y_true, np.array(preds_onnx))
        lang_accs = per_language_accuracy(preds_onnx, test_df)
        ratio = round(full_size_kb / onnx_size_kb, 2)
        print(f"  Accuracy: {m['Accuracy (%)']:.2f}%  Size: {onnx_size_kb:.0f} KB  "
              f"Inference: {inf_ms_onnx:.4f} ms  Ratio: {ratio}x")
        all_results.append({"Version": "ONNX", "Size_KB": round(onnx_size_kb, 1),
                             **m, "Inference_Time_ms": round(inf_ms_onnx, 4),
                             "Compression_Ratio": ratio,
                             **{f"{k}_Acc": v for k, v in lang_accs.items()}})

    # ── Version 3: Quantized ONNX ──
    print("\n  [3/3] Dynamic INT8 quantization...")
    ok = quantize_onnx(onnx_path, quant_path)
    if ok and os.path.exists(quant_path):
        quant_size_kb = os.path.getsize(quant_path) / 1024
        preds_q, inf_ms_q = onnx_predict_all(quant_path, tokenizer, sentences)
        if preds_q:
            m = all_classification_metrics(y_true, np.array(preds_q))
            lang_accs = per_language_accuracy(preds_q, test_df)
            ratio = round(full_size_kb / quant_size_kb, 2)
            print(f"  Accuracy: {m['Accuracy (%)']:.2f}%  Size: {quant_size_kb:.0f} KB  "
                  f"Inference: {inf_ms_q:.4f} ms  Ratio: {ratio}x")
            all_results.append({"Version": "ONNX Quantized (QInt8)", "Size_KB": round(quant_size_kb, 1),
                                 **m, "Inference_Time_ms": round(inf_ms_q, 4),
                                 "Compression_Ratio": ratio,
                                 **{f"{k}_Acc": v for k, v in lang_accs.items()}})

    results_df = pd.DataFrame(all_results)
    out_path = f"{RESULTS_DIR}/nlp_compression_comparison.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\n  Saved: {out_path}")
    print("\n  === NLP Compression Summary ===")
    print(results_df[["Version", "Size_KB", "Accuracy (%)", "Inference_Time_ms", "Compression_Ratio"]].to_string(index=False))
    print("\n  [DONE] MuRIL compression complete.")


if __name__ == "__main__":
    main()
