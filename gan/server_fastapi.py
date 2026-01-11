import os
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, conlist

from inference_cached import WaveGANService, load_service_from_env, _write_wav

__author__ = "Riccardo Petrini"

app = FastAPI(title="")

OUTPUT_PATH = "static/output.wav"
EVOLUTION_DIR = "static/evolution"
BATCH_DIR = "static/batch"

os.makedirs("static", exist_ok=True)
os.makedirs(EVOLUTION_DIR, exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)

_service: WaveGANService = load_service_from_env()
_asr = None


class GenerateBody(BaseModel):
    categorical_code: conlist(int, min_items=16, max_items=16)


class LayerBody(GenerateBody):
    layer: str


class TranscribeBody(BaseModel):
    file: str
    language: Optional[str] = None


def _check_code(code: list[int]):
    if len(code) != 16 or any(v not in (0, 1) for v in code):
        raise HTTPException(status_code=400, detail="Provide 16 binary values in 'categorical_code'")


def _asr_model():
    global _asr
    if _asr is None:
        from faster_whisper import WhisperModel

        name = os.environ.get("WHISPER_MODEL", "small")
        device = os.environ.get("WHISPER_DEVICE", "cpu")
        compute = os.environ.get("WHISPER_COMPUTE", "int8")
        _asr = WhisperModel(name, device=device, compute_type=compute)
    return _asr


@app.get("/")
def root():
    return {"status": "ok", "author": __author__}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate")
def generate(body: GenerateBody):
    _check_code(body.categorical_code)
    z = np.random.uniform(-1, 1, (1, 100)).astype(np.float32)
    z[0, :16] = np.array(body.categorical_code, dtype=np.float32)
    audio = _service.generate(z)
    _write_wav(audio, OUTPUT_PATH)
    return FileResponse(OUTPUT_PATH, media_type="audio/wav")


@app.post("/generate_file")
def generate_file(body: GenerateBody):
    _check_code(body.categorical_code)
    path, z = _service.synthesize_and_store(body.categorical_code, BATCH_DIR)
    return {"file": "/" + path.replace("\\", "/"), "z": z.tolist(), "code": body.categorical_code}


@app.post("/generate_evolution")
def generate_evolution(body: GenerateBody):
    _check_code(body.categorical_code)
    z = np.random.uniform(-1, 1, (1, 100)).astype(np.float32)
    z[0, :16] = np.array(body.categorical_code, dtype=np.float32)
    layers, final_len = _service.generate_layers(z, EVOLUTION_DIR)
    payload = []
    for layer in layers:
        payload.append(
            {
                "name": layer["name"],
                "raw_url": "/" + os.path.relpath(layer["raw_path"], start="static").replace("\\", "/"),
                "stretched_url": "/" + os.path.relpath(layer["stretched_path"], start="static").replace("\\", "/"),
                "raw_len": layer["raw_len"],
                "stretched_len": layer["stretched_len"],
            }
        )
    return {"final_len": final_len, "layers": payload}


@app.post("/generate_layer_file")
def generate_layer_file(body: LayerBody):
    _check_code(body.categorical_code)
    target = (body.layer or "").lower()
    if not target:
        raise HTTPException(status_code=400, detail="Provide 'layer'")

    z = np.random.uniform(-1, 1, (1, 100)).astype(np.float32)
    z[0, :16] = np.array(body.categorical_code, dtype=np.float32)
    layers, _ = _service.generate_layers(z, os.path.join(EVOLUTION_DIR, f"req_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"))
    hit = next((l for l in layers if (l["name"] or "").lower() == target), None)
    if not hit:
        raise HTTPException(status_code=404, detail=f"Layer '{target}' not found")

    fname = f"layer_{target}_{uuid.uuid4().hex[:6]}.wav"
    dst = os.path.join(BATCH_DIR, fname)
    os.makedirs(BATCH_DIR, exist_ok=True)
    with open(hit["stretched_path"], "rb") as src, open(dst, "wb") as out:
        out.write(src.read())

    resp = {"file": "/" + dst.replace("\\", "/"), "categorical_code": body.categorical_code}
    return resp


@app.post("/transcribe_file")
def transcribe_file(body: TranscribeBody):
    path = body.file.lstrip("/") if body.file else None
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"file not found: {body.file}")

    model = _asr_model()
    segments, info = model.transcribe(path, language=body.language or None, vad_filter=True, beam_size=5, word_timestamps=True)

    out = {"language": info.language, "duration": float(info.duration), "text": "", "segments": [], "words": []}
    texts = []
    for seg in segments:
        entry = {"start": float(seg.start or 0.0), "end": float(seg.end or 0.0), "text": (seg.text or "").strip()}
        if getattr(seg, "words", None):
            words = []
            for w in seg.words:
                if w.start is None or w.end is None:
                    continue
                words.append({"start": float(w.start), "end": float(w.end), "text": getattr(w, "word", getattr(w, "text", "")).strip()})
            entry["words"] = words
            out["words"].extend(words)
        if entry["text"]:
            texts.append(entry["text"])
        out["segments"].append(entry)
    out["text"] = " ".join(texts).strip()
    return out


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server_fastapi:app", host="0.0.0.0", port=port, reload=False)
