from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)
CORS(app)   # ✅ ALLOW Chrome Extension & browser requests

# Prefer loading the full pipeline if available (saved as spam_pipeline.pkl)
try:
    model = pickle.load(open("spam_pipeline.pkl", "rb"))
    using_pipeline = True
except Exception:
    # Fallback to legacy model + vectorizer
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    using_pipeline = False

# Use shared preprocess implementation (required by pickled pipeline)
from utils import preprocess

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Backend running",
        "message": "Spam Detection API is active",
        "using_pipeline": using_pipeline
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Message field is required"}), 400

    message = data["message"]

    if using_pipeline:
        prob = float(model.predict_proba([message])[0][1])
        pred = "Spam" if prob >= 0.5 else "Not Spam"
        return jsonify({
            "input_message": message,
            "prediction": pred,
            "probability": round(prob, 4)
        })
    else:
        processed = preprocess(message)
        vector = vectorizer.transform([processed])
        prob_arr = model.predict_proba(vector)[0]
        pred = "Spam" if model.predict(vector)[0] == 1 else "Not Spam"
        return jsonify({
            "input_message": message,
            "prediction": pred,
            "probability": round(float(max(prob_arr)), 4)
        })

if __name__ == "__main__":
    # disable reloader to avoid socket.fromfd OSError on Windows
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
