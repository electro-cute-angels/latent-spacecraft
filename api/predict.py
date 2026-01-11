"""Cog predictor for the concatenation GAN.
Author: Riccardo Petrini
"""
from typing import List
from cog import BasePredictor, Input, Path

__author__ = "Riccardo Petrini"
_MODEL = None


class Predictor(BasePredictor):
    def setup(self) -> None:
        # Intentionally lazy to avoid pulling TF during schema generation
        pass

    def predict(
        self,
        z: List[float] = Input(
            description="100-D latent vector in [-1, 1]."
        ),
        sample_rate: int = Input(
            default=16000,
            description="Output WAV sample rate (model trained at 16 kHz).",
        ),
    ) -> Path:
        import numpy as np
        import soundfile as sf

        global _MODEL
        if _MODEL is None:
            from model import load_model

            _MODEL = load_model()

        z_vec = np.asarray(z, dtype=np.float32)
        if z_vec.shape != (100,):
            raise ValueError(f"Expected 100 floats for z, got shape {z_vec.shape}.")

        audio = _MODEL.generate_audio(z_vec.reshape(1, 100))

        out_path = Path("out.wav")
        sf.write(str(out_path), audio, samplerate=sample_rate, format="WAV")
        return out_path
