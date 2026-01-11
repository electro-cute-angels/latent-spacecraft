import os
import time
import uuid
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from inference_cached import WaveGANService, load_service_from_env, _write_wav

__author__ = "Riccardo Petrini"

app = Flask(__name__)
CORS(app)

OUTPUT_PATH = "static/output.wav"
BATCH_DIR = "static/batch"
EVOLUTION_DIR = "static/evolution"

os.makedirs("static", exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)
os.makedirs(EVOLUTION_DIR, exist_ok=True)

service: WaveGANService = load_service_from_env()


def _code(payload):
    code = payload.get("categorical_code", []) if payload else []
    if len(code) != 16 or any(v not in (0, 1) for v in code):
        return None
    return code


@app.route("/", methods=["GET"])
def root():
    return {"author": __author__, "status": "ready"}


@app.route("/generate", methods=["POST"])
def generate():
    code = _code(request.json)
    if not code:
        return jsonify({"error": "Provide 16 binary values in 'categorical_code'"}), 400
    z = np.random.uniform(-1, 1, (1, 100)).astype(np.float32)
    z[0, :16] = np.array(code, dtype=np.float32)
    audio = service.generate(z)
    _write_wav(audio, OUTPUT_PATH)
    return send_file(OUTPUT_PATH, mimetype="audio/wav")


@app.route("/generate_file", methods=["POST"])
def generate_file():
    code = _code(request.json)
    if not code:
        return jsonify({"error": "Provide 16 binary values in 'categorical_code'"}), 400
    path, z = service.synthesize_and_store(code, BATCH_DIR)
    return jsonify({"file": "/" + path.replace("\\", "/"), "code": code, "z": z.tolist()})


@app.route("/layers", methods=["POST"])
def layers():
    code = _code(request.json)
    if not code:
        return jsonify({"error": "Provide 16 binary values in 'categorical_code'"}), 400
    z = np.random.uniform(-1, 1, (1, 100)).astype(np.float32)
    z[0, :16] = np.array(code, dtype=np.float32)
    req_dir = os.path.join(EVOLUTION_DIR, f"req_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}")
    layers, final_len = service.generate_layers(z, req_dir)
    payload = []
    for layer in layers:
        payload.append(
            {
                "name": layer["name"],
                "raw": "/" + os.path.relpath(layer["raw_path"], start="static").replace("\\", "/"),
                "stretched": "/" + os.path.relpath(layer["stretched_path"], start="static").replace("\\", "/"),
                "raw_len": layer["raw_len"],
                "stretched_len": layer["stretched_len"],
            }
        )
    return jsonify({"final_len": final_len, "layers": payload})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
