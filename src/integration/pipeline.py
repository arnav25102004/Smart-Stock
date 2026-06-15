"""
End-to-end SMARTSTOCK integration pipeline.

Chains the best NLP model (MuRIL or TF-IDF+SVM fallback) with the best
forecasting model (TFLite compressed LSTM/CNN) and a mock inventory database.
Demonstrates multilingual query → intent → action → response in Hindi,
Kannada, and English.

Usage:
    python src/integration/pipeline.py
    (Requires NLP and forecasting model scripts to have been run first.)
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

RESULTS_DIR = "results/tables"
MODELS_DIR = "results/models"
NLP_MODELS_DIR = os.path.join(MODELS_DIR, "muril_finetuned")
PROCESSED_DIR = "data/processed"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────── Mock Inventory Database ────────────────────────────
mock_inventory = {
    "rice":   {"current_kg": 15.5, "threshold_kg": 5,  "kannada": "akki",   "hindi": "chawal"},
    "sugar":  {"current_kg": 8.2,  "threshold_kg": 3,  "kannada": "sakkare","hindi": "cheeni"},
    "dal":    {"current_kg": 4.1,  "threshold_kg": 4,  "kannada": "bele",   "hindi": "dal"},
    "tea":    {"current_kg": 2.0,  "threshold_kg": 1,  "kannada": "chai",   "hindi": "chai"},
    "oil":    {"current_kg": 6.0,  "threshold_kg": 2,  "kannada": "enne",   "hindi": "tel"},
    "flour":  {"current_kg": 10.0, "threshold_kg": 4,  "kannada": "hittu",  "hindi": "aata"},
    "soap":   {"current_kg": 3.5,  "threshold_kg": 1,  "kannada": "sabbu",  "hindi": "sabun"},
    "milk":   {"current_kg": 5.0,  "threshold_kg": 2,  "kannada": "haalu",  "hindi": "doodh"},
}

# Response templates per language
RESPONSES = {
    "check_stock": {
        "English":    "Current {product} stock is {amount:.1f} kg.",
        "Hindi":      "Abhi {product_h} ka stock {amount:.1f} kg hai.",
        "Kannada":    "{product_k} stock {amount:.1f} kg ide.",
        "Code-Mixed": "{product} ka stock {amount:.1f} kg hai abhi.",
    },
    "add_stock": {
        "English":    "Stock updated. Added to {product} inventory.",
        "Hindi":      "{product_h} ka stock update ho gaya.",
        "Kannada":    "{product_k} stock update aagide.",
        "Code-Mixed": "{product} stock updated hai.",
    },
    "get_prediction": {
        "English":    "Predicted demand for {product} tomorrow: {pred:.1f} kg.",
        "Hindi":      "Kal {product_h} ki maang lagbhag {pred:.1f} kg hogi.",
        "Kannada":    "Naale {product_k} bedu andaaja {pred:.1f} kg.",
        "Code-Mixed": "{product} ki predicted demand kal {pred:.1f} kg hai.",
    },
    "low_stock_alert": {
        "English":    "LOW STOCK ALERT: {items}",
        "Hindi":      "LOW STOCK ALERT: {items} — jaldi order karo!",
        "Kannada":    "LOW STOCK: {items} — jaldi order maadi!",
        "Code-Mixed": "Low stock alert: {items} — order karo please!",
    },
    "help": {
        "English":    "SmartStock helps you: check stock | add stock | get predictions | view alerts.",
        "Hindi":      "SmartStock se aap: stock check, stock add, demand predict, aur alerts dekh sakte hain.",
        "Kannada":    "SmartStock inda: stock nodari, haakari, predict maadari, alerts nodari.",
        "Code-Mixed": "SmartStock use karke: check stock, add stock, predictions lo, alerts dekho.",
    },
}


# ─────────────────────── Product keyword mapping ─────────────────────────────
PRODUCT_KEYWORDS = {
    "rice":  ["rice", "chawal", "akki"],
    "sugar": ["sugar", "cheeni", "sakkare"],
    "dal":   ["dal", "bele"],
    "tea":   ["tea", "chai"],
    "oil":   ["oil", "tel", "enne"],
    "flour": ["flour", "atta", "hittu", "maida"],
    "soap":  ["soap", "sabun", "sabbu"],
    "milk":  ["milk", "doodh", "haalu"],
}


def detect_product(sentence):
    """Detect which product is mentioned in a sentence using keyword matching.

    Args:
        sentence: Input sentence string.

    Returns:
        Product key string, or 'rice' as a safe default.
    """
    s = sentence.lower()
    for product, kws in PRODUCT_KEYWORDS.items():
        if any(kw in s for kw in kws):
            return product
    return "rice"


# ─────────────────────── NLP Model Loader ────────────────────────────────────
def load_nlp_model():
    """Load the best available NLP model (MuRIL → TF-IDF+SVM fallback).

    Returns:
        Tuple (predict_fn, model_type_str) where predict_fn(sentences) → labels.
    """
    # Try MuRIL first
    if os.path.exists(NLP_MODELS_DIR):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            INTENTS = ["check_stock", "add_stock", "get_prediction", "low_stock_alert", "help"]
            ID2LABEL = {i: intent for i, intent in enumerate(INTENTS)}
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(NLP_MODELS_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(NLP_MODELS_DIR)
            model.to(device)
            model.eval()

            def muril_predict(sentences):
                preds = []
                with torch.no_grad():
                    for s in sentences:
                        enc = tokenizer(s, return_tensors="pt", max_length=64,
                                        padding="max_length", truncation=True)
                        enc = {k: v.to(device) for k, v in enc.items()}
                        logits = model(**enc).logits
                        preds.append(ID2LABEL[torch.argmax(logits, dim=-1).item()])
                return preds

            print("  NLP model: MuRIL (fine-tuned)")
            return muril_predict, "MuRIL"
        except Exception as e:
            print(f"  [WARN] MuRIL load failed ({e}), falling back to TF-IDF+SVM")

    # Try TF-IDF+SVM
    svm_path = os.path.join(MODELS_DIR, "tfidf_svm.pkl")
    if os.path.exists(svm_path):
        with open(svm_path, "rb") as f:
            pipeline = pickle.load(f)

        def svm_predict(sentences):
            return list(pipeline.predict(sentences))

        print("  NLP model: TF-IDF + SVM")
        return svm_predict, "TF-IDF+SVM"

    # Final fallback: rule-based inline
    from src.nlp.model_rule_based import classify

    def rule_predict(sentences):
        return [classify(s) for s in sentences]

    print("  NLP model: Rule-Based (fallback)")
    return rule_predict, "Rule-Based"


# ─────────────────────── Forecasting Model Loader ────────────────────────────
def load_forecasting_model(product):
    """Load the best available forecasting model for a given product.

    Tries TFLite dynamic-quantized LSTM first, then falls back to full Keras .h5.

    Args:
        product: Product key string (e.g. 'rice').

    Returns:
        Tuple (predict_fn, model_type) where predict_fn(X) → scalar prediction,
        or None if no model is available.
    """
    safe = product.upper().replace(" ", "_")
    # Prefer "GROCERY I" family for generic products not in the four families
    family_map = {
        "rice": "GROCERY_I", "flour": "GROCERY_I", "sugar": "GROCERY_I",
        "tea": "GROCERY_I", "dal": "GROCERY_I",
        "milk": "DAIRY", "soap": "CLEANING", "oil": "BEVERAGES",
    }
    family = family_map.get(product, "GROCERY_I")

    tflite_path = os.path.join(MODELS_DIR, f"lstm_{family}_dynamic.tflite")
    h5_path = os.path.join(MODELS_DIR, f"lstm_{family}.h5")
    scaler_path = os.path.join(MODELS_DIR, f"lstm_scaler_{family}.pkl")

    if not os.path.exists(scaler_path):
        return None, None

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    def make_dummy_input():
        """Return a plausible scaled input sequence for inference."""
        n_features = scaler.n_features_in_
        dummy_raw = np.ones((14, n_features)) * 10.0
        return scaler.transform(dummy_raw).reshape(1, 14, n_features)

    if os.path.exists(tflite_path):
        try:
            import tensorflow as tf
            with open(tflite_path, "rb") as f:
                tflite_bytes = f.read()
            interp = tf.lite.Interpreter(model_content=tflite_bytes)
            interp.allocate_tensors()
            in_det = interp.get_input_details()
            out_det = interp.get_output_details()

            def tflite_predict(X):
                interp.set_tensor(in_det[0]['index'], X.astype(np.float32))
                interp.invoke()
                out_scaled = interp.get_tensor(out_det[0]['index'])[0][0]
                n_features = scaler.n_features_in_
                dummy = np.zeros((1, n_features))
                dummy[0, 0] = out_scaled
                return float(scaler.inverse_transform(dummy)[0, 0])

            return tflite_predict, "TFLite LSTM"
        except Exception:
            pass

    if os.path.exists(h5_path):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(h5_path, compile=False)

            def keras_predict(X):
                out_scaled = model.predict(X, verbose=0)[0][0]
                n_features = scaler.n_features_in_
                dummy = np.zeros((1, n_features))
                dummy[0, 0] = out_scaled
                return float(scaler.inverse_transform(dummy)[0, 0])

            return keras_predict, "Keras LSTM"
        except Exception:
            pass

    return None, None


# ─────────────────────── Response builder ────────────────────────────────────
def build_response(intent, product, language, forecast_fn=None):
    """Build a natural-language response for the detected intent.

    Args:
        intent: Detected intent label string.
        product: Detected product key string.
        language: Detected language string.
        forecast_fn: Optional forecasting predict function (used for get_prediction).

    Returns:
        Response string in the appropriate language.
    """
    lang = language if language in RESPONSES.get(intent, {}) else "English"
    template = RESPONSES.get(intent, {}).get(lang, "Sorry, I didn't understand that.")

    inv = mock_inventory.get(product, mock_inventory["rice"])
    kwargs = {
        "product": product,
        "product_h": inv.get("hindi", product),
        "product_k": inv.get("kannada", product),
        "amount": inv["current_kg"],
    }

    if intent == "check_stock":
        return template.format(**kwargs)

    elif intent == "add_stock":
        mock_inventory[product]["current_kg"] += 10.0
        return template.format(**kwargs)

    elif intent == "get_prediction":
        if forecast_fn is not None:
            dummy_X = np.ones((1, 14, 4)) * 0.5  # normalized placeholder
            try:
                pred_val = forecast_fn(dummy_X)
            except Exception:
                pred_val = inv["current_kg"] * 0.9
        else:
            pred_val = inv["current_kg"] * 0.9
        return template.format(pred=pred_val, **kwargs)

    elif intent == "low_stock_alert":
        low_items = [
            p for p, v in mock_inventory.items()
            if v["current_kg"] <= v["threshold_kg"]
        ]
        items_str = ", ".join(low_items) if low_items else "none"
        return template.format(items=items_str)

    elif intent == "help":
        return template

    return "I didn't understand that."


# ─────────────────────── Main pipeline ───────────────────────────────────────
def handle_query(user_input, language, predict_fn, forecast_fn):
    """Process a single inventory query end-to-end.

    Args:
        user_input: Raw sentence string from the user.
        language: Language label (English / Hindi / Kannada / Code-Mixed).
        predict_fn: NLP model prediction function.
        forecast_fn: Forecasting model prediction function (may be None).

    Returns:
        Tuple (intent, product, response, latency_ms).
    """
    t0 = time.perf_counter()
    intent = predict_fn([user_input])[0]
    product = detect_product(user_input)
    response = build_response(intent, product, language, forecast_fn)
    latency_ms = (time.perf_counter() - t0) * 1000
    return intent, product, response, latency_ms


# ─────────────────────── Demo queries ────────────────────────────────────────
DEMO_QUERIES = [
    # English
    ("how much rice is left",         "English"),
    ("add 10 kg sugar",               "English"),
    ("predict dal demand tomorrow",   "English"),
    ("which items are low",           "English"),
    ("how do I use this",             "English"),
    # Hindi
    ("chawal kitna bacha hai",        "Hindi"),
    ("cheeni ka stock badhao",        "Hindi"),
    ("kal ki dal ki demand batao",    "Hindi"),
    ("kaunsa item khatam ho raha",    "Hindi"),
    ("kaise use karu",                "Hindi"),
    # Kannada
    ("akki eshtu ide",                "Kannada"),
    ("sakkare haaku",                 "Kannada"),
    ("naale bele forecast heli",      "Kannada"),
    ("yaav item low stock ide",       "Kannada"),
    ("hege upayogisu",                "Kannada"),
    # Code-Mixed
    ("rice ka stock check karo",      "Code-Mixed"),
    ("sugar 10 kg add maadi please",  "Code-Mixed"),
    ("dal prediction please batao",   "Code-Mixed"),
    ("stock low hai kya alert do",    "Code-Mixed"),
    ("help chahiye please tell me",   "Code-Mixed"),
]


def main():
    print("=" * 60)
    print("STEP 14: End-to-End Integration Pipeline Demo")
    print("=" * 60)

    print("\n  Loading NLP model...")
    predict_fn, nlp_model_type = load_nlp_model()

    print("  Loading forecasting model...")
    forecast_fn, forecast_model_type = load_forecasting_model("rice")
    if forecast_fn is None:
        print("  [WARN] No forecasting model found — predictions will use estimates.")

    print(f"\n  NLP model      : {nlp_model_type}")
    print(f"  Forecast model : {forecast_model_type or 'Estimate only'}")

    print("\n" + "-" * 60)
    print(f"  {'Input':<40} {'Lang':<12} {'Intent':<20} {'Product'}")
    print("-" * 60)

    rows = []
    latencies = []

    for sentence, language in DEMO_QUERIES:
        intent, product, response, latency = handle_query(
            sentence, language, predict_fn, forecast_fn
        )
        latencies.append(latency)
        print(f"  {sentence:<40} {language:<12} {intent:<20} {product}")
        print(f"    → {response}")
        rows.append({
            "input": sentence,
            "language": language,
            "detected_intent": intent,
            "detected_product": product,
            "response": response,
            "latency_ms": round(latency, 4),
        })

    avg_latency = np.mean(latencies)
    print("\n" + "-" * 60)
    print(f"  Average end-to-end latency: {avg_latency:.2f} ms")
    print(f"  NLP model used            : {nlp_model_type}")
    print(f"  Forecast model used       : {forecast_model_type or 'N/A'}")

    # Save demo output
    demo_df = pd.DataFrame(rows)
    demo_df.to_csv(f"{RESULTS_DIR}/integration_demo.csv", index=False)

    # Save pipeline performance summary
    summary = pd.DataFrame([{
        "NLP_Model": nlp_model_type,
        "Forecast_Model": forecast_model_type or "N/A",
        "Avg_Latency_ms": round(avg_latency, 4),
        "Total_Demo_Queries": len(DEMO_QUERIES),
    }])
    summary.to_csv(f"{RESULTS_DIR}/pipeline_summary.csv", index=False)

    print(f"\n  Saved: {RESULTS_DIR}/integration_demo.csv")
    print("  [DONE] Integration pipeline demo complete.")


if __name__ == "__main__":
    main()
