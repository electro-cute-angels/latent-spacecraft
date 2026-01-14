import os
import numpy as np
import torch
import scipy.io.wavfile
from typing import List, Dict, Tuple

from infowavegan import WaveGANGenerator

SAMPLE_RATE = 16000
SLICE_LEN = 65536


def _write_wav(arr: np.ndarray, path: str, normalize: bool = True) -> None:
    x = arr.astype(np.float32)
    if normalize:
        m = np.max(np.abs(x))
        if m > 0:
            x = x / m
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scipy.io.wavfile.write(path, SAMPLE_RATE, x)


def _resample_to_len(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[0] == target_len:
        return x.astype(np.float32)
    t_src = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=False)
    t_dst = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(t_dst, t_src, x).astype(np.float32)


def load_generator(ckpt_path: str, slice_len: int = SLICE_LEN) -> Tuple[WaveGANGenerator, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = WaveGANGenerator(slice_len=slice_len).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(state)
    return G, device


def generate_layers(
    z: np.ndarray,
    generator: WaveGANGenerator,
    device: torch.device,
    output_dir: str,
) -> Tuple[List[Dict[str, object]], int]:
    """
    Run one forward pass, capture upconv layers, and write raw and stretched WAVs.
    Returns (layers, final_len) where layers is a list of dicts with name/raw_path/stretched_path/raw_len/stretched_len.
    """
    os.makedirs(output_dir, exist_ok=True)
    z_t = torch.from_numpy(z).to(torch.float32).to(device)

    captured: Dict[str, torch.Tensor] = {}

    def hook(name: str):
        def _fn(module, inp, out):
            captured[name] = out.detach()
        return _fn

    hooks = []
    try:
        hooks += [generator.z_batchnorm.register_forward_hook(hook("z_project"))]
        hooks += [generator.upconv0.register_forward_hook(hook("upconv0"))]
        hooks += [generator.upconv1.register_forward_hook(hook("upconv1"))]
        hooks += [generator.upconv2.register_forward_hook(hook("upconv2"))]
        hooks += [generator.upconv3.register_forward_hook(hook("upconv3"))]
        hooks += [generator.upconv4.register_forward_hook(hook("upconv4"))]
        if hasattr(generator, "upconv5"):
            hooks += [generator.upconv5.register_forward_hook(hook("upconv5"))]

        final_t = generator(z_t).detach().cpu()  # (1,1,65536)
    finally:
        for h in hooks:
            h.remove()

    final = final_t.numpy()[0, 0, :].astype(np.float32)
    final_len = final.shape[0]

    order = ["upconv1", "upconv2", "upconv3", "upconv4"]
    if "upconv5" in captured:
        order.append("upconv5")

    results: List[Dict[str, object]] = []
    for name in order:
        if name not in captured:
            continue
        t = captured[name][0, 0, :].cpu().numpy().astype(np.float32)
        raw_path = os.path.join(output_dir, f"{name}_raw.wav")
        stretched_path = os.path.join(output_dir, f"{name}_stretched.wav")

        _write_wav(t, raw_path, normalize=True)
        _write_wav(_resample_to_len(t, final_len), stretched_path, normalize=True)

        results.append({
            "name": name,
            "raw_path": raw_path,
            "stretched_path": stretched_path,
            "raw_len": int(t.shape[0]),
            "stretched_len": int(final_len),
        })

    return results, final_len
