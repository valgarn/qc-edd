import os
import logging
import threading
import base64
from io import BytesIO
from typing import Optional, TypedDict, Any, Tuple

from flask import Flask, request, jsonify
from flask.typing import ResponseReturnValue
from flask_cors import CORS

MAX_AUDIO_BYTES: int = 25 * 1024 * 1024  # 25 MB

def create_rest_app() -> Flask:
    app: Flask = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    @app.route('/qc-edd/parkinson', methods=['POST'])
    def parse_form() -> ResponseReturnValue:
        data_raw: Optional[dict[str, Any]] = request.get_json(silent=True)
        data: ParkinsonRequest = {} if data_raw is None else data_raw  # type: ignore[assignment]

        audio_b64: Optional[str] = data.get("audio")
        gender: Optional[str] = data.get("gender")
        age: Optional[int] = data.get("age")  # or Optional[str] if clients send it as a string

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

        audio_buffer: BytesIO = BytesIO(audio_bytes)

        try:
            with open("audio_input.wav", "wb") as f:
                f.write(audio_bytes)
        except Exception as e:
            app.logger.error(f"Write failure: {e}")
            return jsonify({"error": "Unable to store audio"}), 500

        # Placeholder inference result
        probability: float = 0.85

        return jsonify({
            "probability": probability,
        }), 200

    return app

def serve_rest() -> None:
    app: Flask = create_rest_app()
    rest_port: int = int(os.environ.get('REST_PORT', '7310'))
    app.logger.info(f"REST server listening on port {rest_port}")
    app.run(host='0.0.0.0', port=rest_port, threaded=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve_rest()
    # If you truly need to run in a thread:
    # rest_thread: threading.Thread = threading.Thread(target=serve_rest)
    # rest_thread.start()
    # rest_thread.join()
