from typing import Optional, TypedDict, Any, Tuple

import os
import logging
import threading
import base64
import json

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

from flask import Flask, request, jsonify
from flask_cors import CORS

import quantum_data
import quantum

import google.cloud.logging

MAX_AUDIO_BYTES: int = 25 * 1024 * 1024  # 25 MB
FEATURES_ORDER = ["Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP","Shimmer", "Shimmer(dB)", 
                    "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"]

def create_rest_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    @app.route('/qc-edd/parkinson', methods=['POST'])
    def predict_parkinson():
        data_raw: dict[str, Any] = request.get_json(silent=True)
        data: ParkinsonRequest = {} if data_raw is None else data_raw  # type: ignore[assignment]

        audio_b64: str = data.get("audio")
        gender: str = data.get("gender")
        age: int = data.get("age")

        if not audio_b64:
            return jsonify({"error": "Audio is required"}), 400

        # Handle possible data URL prefix
        if audio_b64.strip().startswith("data:") and "," in audio_b64:
            audio_b64 = audio_b64.split(",", 1)[1]

        try:
            audio_bytes: bytes = base64.b64decode(audio_b64, validate=True)
        except Exception:
            app.logger.warning("Invalid base64 audio input")
            return jsonify({"error": "Invalid audio encoding"}), 400

        if len(audio_bytes) > MAX_AUDIO_BYTES:
            return jsonify({"error": "Audio too large"}), 413
        try:
            with open("audio_input.wav", "wb") as f:
                f.write(audio_bytes)
        except Exception as e:
            # The line `app.logger.error(f"Write failure: {e}")` is logging an error message using the
            # logger associated with the Flask application (`app`).
            app.logger.error(f"Write failure: {e}")
            return jsonify({"error": "Unable to store audio"}), 500

        feats = quantum_data.compute_voice_features_from_bytes(audio_bytes)
        print("feats: ", feats)
        X = pd.DataFrame(np.array([[age, 1 if gender == 'female' else 0, 1] + [feats[name] for name in FEATURES_ORDER]], dtype=np.float32))
        X = X[quantum_data.FEATURES_SIGNIFICANT]
        X = X.values
        print("X: ", X)
        probability = quantum.predict(X, quantum_data.X_SCALER, quantum_data.Y_SCALER).tolist()
        return jsonify({
            "probability": quantum_data.parkinson_probability(probability[0][0], probability[0][1], method="hybrid"),
        }), 200

    return app


def serve_rest():
    app = create_rest_app()
    rest_port = os.environ.get('REST_PORT', '7311')
    print(f"REST server started, listening on port {rest_port}")
    app.run(host='0.0.0.0', port=int(rest_port))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    X, Y, dataset = quantum_data.init_foundation_data()
    #probability = quantum.predict(X, quantum_data.X_SCALER, quantum_data.Y_SCALER)
    #print(probability)
    serve_rest()
    
    #client = google.cloud.logging.Client()
    #handler = client.get_default_handler()
    #logging.getLogger().setLevel(logging.INFO)
    #logging.getLogger().addHandler(handler)
    #rest_thread = threading.Thread(target=serve_rest, daemon=True)
    #rest_thread.start()
    #rest_thread.join()
