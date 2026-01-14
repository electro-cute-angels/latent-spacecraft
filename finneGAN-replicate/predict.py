from typing import Dict, Any, Optional
import os
import tempfile
import numpy as np
import torch
from cog import BasePredictor, Path, Input
import whisper

from inference import load_generator, generate_layers, SAMPLE_RATE, SLICE_LEN


class Predictor(BasePredictor):
    def setup(self) -> None:

        ckpt_env = os.environ.get("FINNEGAN_CHECKPOINT", "checkpoint/epoch450_step166500_G.pt")
        self.generator, self.device = load_generator(ckpt_env, slice_len=SLICE_LEN)

        
        whisper_name = os.environ.get("WHISPER_MODEL", "small")
        whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper = whisper.load_model(whisper_name, device=whisper_device)

    def predict(
        self,
        seed: Optional[int] = Input(
            default=None,
            description="Optional RNG seed for reproducibility of latent and categorical code.",
        ),
    ) -> Any:
        rng = np.random.default_rng(seed)

        
        categorical = rng.integers(low=0, high=2, size=(16,), dtype=np.int32)
        z = rng.uniform(-1.0, 1.0, size=(1, 100)).astype(np.float32)
        z[0, :16] = categorical.astype(np.float32)

        out_dir = tempfile.mkdtemp(prefix="finnegan_layers_")
        layers, final_len = generate_layers(z, self.generator, self.device, out_dir)

        layer_paths: Dict[str, Dict[str, Path]] = {}
        for layer in layers:
            name = layer["name"]
            layer_paths[name] = {
                "raw": Path(layer["raw_path"]),
                "stretched": Path(layer["stretched_path"]),
            }

        transcription = self._transcribe(layer_paths.get("upconv5", {}).get("stretched"))

        return {
            "latent": z[0].tolist(),
            "categorical_code": categorical.tolist(),
            "sample_rate": SAMPLE_RATE,
            "final_len": final_len,
            "layers": layer_paths,
            "transcription": transcription,
        }

    def _transcribe(self, stretched_path: Optional[Path]) -> Dict[str, Any]:
        if stretched_path is None:
            return {}

        result = self.whisper.transcribe(
            str(stretched_path),
            language=None,
            word_timestamps=True,
            verbose=False,
        )

        out: Dict[str, Any] = {
            "language": result.get("language"),
            "duration": float(result.get("duration", 0.0)),
            "text": result.get("text", "").strip(),
            "segments": [],
            "words": [],
        }

        for seg in result.get("segments", []) or []:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", s))
            txt = (seg.get("text") or "").strip()
            seg_entry: Dict[str, Any] = {"start": s, "end": e, "text": txt}
            wlist = []
            for w in seg.get("words", []) or []:
                w_item = {
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                    "text": (w.get("word") or w.get("text") or "").strip(),
                }
                wlist.append(w_item)
                out["words"].append(w_item)
            if wlist:
                seg_entry["words"] = wlist
            out["segments"].append(seg_entry)
        return out
