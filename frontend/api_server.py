from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Ensure the src folder is accessible for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)
# Enable Cross-Origin Resource Sharing so the separate HTML file can make requests
CORS(app)

# Attempt to load the actual NLP pipeline if available, otherwise use a smart mock
try:
    from src.integration.pipeline import NLPPipeline
    # If the class actually requires initialization, do it here
    nlp_system = NLPPipeline()
    USE_MOCK = False
except Exception as e:
    print(f"Warning: Could not load the real NLP pipeline ({e}). Using fallback logic.")
    USE_MOCK = True

@app.route('/query', methods=['POST'])
def query_inventory():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text'].strip()

    if USE_MOCK:
        # Fallback Mock Logic to satisfy the dashboard UI demo requirement
        text_lower = text.lower()
        
        # Simple heuristic mapping for the mock
        if "chawal" in text_lower or "rice" in text_lower or "akki" in text_lower:
            product = "rice"
            amount = "15.5 kg"
        elif "dal" in text_lower:
            product = "dal"
            amount = "4.1 kg"
        elif "sugar" in text_lower:
            product = "sugar"
            amount = "8.2 kg"
        else:
            product = "unknown"
            amount = "?"

        # Intent heuristic
        if "prediction" in text_lower or "forecast" in text_lower or "tomorrow" in text_lower:
            intent = "demand_forecast"
            resp = f"Predicted demand for {product} is high tomorrow."
        else:
            intent = "check_stock"
            if product != "unknown":
                resp = f"Current stock for {product} is {amount}."
            else:
                resp = "I'm not sure which product you are asking about."

        # Language heuristic
        if "kitna" in text_lower or "hai" in text_lower or "batao" in text_lower:
            lang = "Hindi"
            if product == "rice": resp = f"Abhi chawal ka stock {amount} hai."
            elif product == "dal": resp = f"Dal ka stock {amount} hai."
        elif "eshtu" in text_lower or "ide" in text_lower:
            lang = "Kannada"
            if product == "rice": resp = f"Akki stock {amount} ide."
        else:
            lang = "English"

        return jsonify({
            "intent": intent,
            "language": lang,
            "product": product,
            "response": resp
        })
    else:
        # If the real backend is hooked up
        try:
            # Assuming the interface for NLPPipeline has a predict/process method
            # Adjust this as per the actual pipeline.py structure
            result = nlp_system.process_query(text)
            return jsonify({
                "intent": result.get("intent", "unknown"),
                "language": result.get("language", "unknown"),
                "product": result.get("product", "unknown"),
                "response": result.get("response", "Real model active")
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting SMARTSTOCK Frontend API Backend on port 5000...")
    app.run(port=5000, debug=True)