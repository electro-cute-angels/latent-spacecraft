import os
import uuid
import numpy as np
import torch
import scipy.io.wavfile
from infowavegan import WaveGANGenerator

__author__ = "Riccardo Petrini"

SAMPLE_RATE = 16000
SLICE_LEN = 65536


def _write_wav(arr: np.ndarray, path: str, normalize: bool = True) -> str:
    x = arr.astype(np.float32)
    if normalize:
        peak = np.max(np.abs(x))
        if peak > 0:
            x = x / peak
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scipy.io.wavfile.write(path, SAMPLE_RATE, x)
    return path


class WaveGANService:
    def __init__(self, ckpt_path: str, slice_len: int = SLICE_LEN, sample_rate: int = SAMPLE_RATE):
        self.ckpt_path = ckpt_path
        self.slice_len = slice_len
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._G = None

    def _lazy_load(self):
        if self._G is None:
            G = WaveGANGenerator(slice_len=self.slice_len).to(self.device).eval()
            G.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
            self._G = G
        return self._G

    def generate(self, z: np.ndarray) -> np.ndarray:
        G = self._lazy_load()
        z_t = torch.from_numpy(z).to(torch.float32).to(self.device)
        y = G(z_t).detach().cpu().numpy()[0, 0, :]
        return y.astype(np.float32)

    def generate_layers(self, z: np.ndarray, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        G = self._lazy_load()
        captured = {}

        def hook(name):
            def _fn(_, __, out):
                captured[name] = out.detach()
            return _fn

        handles = []
        try:
            handles.append(G.z_batchnorm.register_forward_hook(hook("z_project")))
            handles.append(G.upconv0.register_forward_hook(hook("upconv0")))
            handles.append(G.upconv1.register_forward_hook(hook("upconv1")))
            handles.append(G.upconv2.register_forward_hook(hook("upconv2")))
            handles.append(G.upconv3.register_forward_hook(hook("upconv3")))
            handles.append(G.upconv4.register_forward_hook(hook("upconv4")))
            if hasattr(G, "upconv5"):
                handles.append(G.upconv5.register_forward_hook(hook("upconv5")))

            final = G(torch.from_numpy(z).to(torch.float32).to(self.device)).detach().cpu()
        finally:
            for h in handles:
                h.remove()

        final_np = final.numpy()[0, 0, :].astype(np.float32)
        final_len = final_np.shape[0]

        order = ["z_project", "upconv0", "upconv1", "upconv2", "upconv3", "upconv4"]
        if "upconv5" in captured:
            order.append("upconv5")

        layers = []
        for name in order:
            arr = captured[name][0, 0, :].cpu().numpy().astype(np.float32)
            raw_path = os.path.join(output_dir, f"{name}_raw.wav")
            stretched_path = os.path.join(output_dir, f"{name}_stretched.wav")
            _write_wav(arr, raw_path)
            stretched = np.interp(
                np.linspace(0.0, 1.0, num=final_len, endpoint=False),
                np.linspace(0.0, 1.0, num=arr.shape[0], endpoint=False),
                arr,
            ).astype(np.float32)
            _write_wav(stretched, stretched_path)
            layers.append(
                {
                    "name": name,
                    "raw_path": raw_path,
                    "stretched_path": stretched_path,
                    "raw_len": int(arr.shape[0]),
                    "stretched_len": int(final_len),
                }
            )

        final_raw = os.path.join(output_dir, "final_raw.wav")
        _write_wav(final_np, final_raw)
        layers.append(
            {
                "name": "final",
                "raw_path": final_raw,
                "stretched_path": final_raw,
                "raw_len": int(final_len),
                "stretched_len": int(final_len),
            }
        )
        return layers, final_len

    def synthesize_and_store(self, code: list[int], batch_dir: str):
        z = np.random.uniform(-1, 1, (1, 100)).astype(np.float32)
        z[0, : len(code)] = np.array(code, dtype=np.float32)
        audio = self.generate(z)
        fname = f"gen_{uuid.uuid4().hex}.wav"
        path = os.path.join(batch_dir, fname)
        _write_wav(audio, path)
        return path, z[0]


def load_service_from_env() -> WaveGANService:
    ckpt = os.environ.get("GAN_CHECKPOINT", "checkpoint/epoch450_step166500_G.pt")
    return WaveGANService(ckpt)
