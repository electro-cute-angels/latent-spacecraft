# Autore: Riccardo Petrini
import os
import numpy as np
import torch
import scipy.io.wavfile
from infowavegan_deterministic import WaveGANGenerator

SAMPLE_RATE = 16000
SLICE_LEN = 65536


def _write_wav(arr: np.ndarray, path: str, normalize: bool = True):
    x = arr.astype(np.float32)
    if normalize:
        m = np.max(np.abs(x))
        if m > 0:
            x = x / m
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scipy.io.wavfile.write(path, SAMPLE_RATE, x)


def _load_generator(ckpt_path: str, slice_len: int = SLICE_LEN, seed: int = 42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = WaveGANGenerator(slice_len=slice_len, seed=seed).to(device).eval()
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    return G, device


def generate_final(z: np.ndarray, ckpt_path: str, seed: int = 42) -> np.ndarray:
    G, device = _load_generator(ckpt_path, slice_len=SLICE_LEN, seed=seed)
    with torch.no_grad():
        z_t = torch.from_numpy(z).to(torch.float32).to(device)
        y = G(z_t).detach().cpu().numpy()[0, 0, :]
    return y.astype(np.float32)


def generate_and_dump(z: np.ndarray, ckpt_path: str, output_dir: str, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    G, device = _load_generator(ckpt_path, slice_len=SLICE_LEN, seed=seed)
    z_t = torch.from_numpy(z).to(torch.float32).to(device)
    captured = {}

    def hook(name):
        def _fn(module, inp, out):
            captured[name] = out.detach().cpu()
        return _fn

    hs = []
    try:
        hs += [G.z_batchnorm.register_forward_hook(hook('z_project'))]
        hs += [G.upconv0.register_forward_hook(hook('upconv0'))]
        hs += [G.upconv1.register_forward_hook(hook('upconv1'))]
        hs += [G.upconv2.register_forward_hook(hook('upconv2'))]
        hs += [G.upconv3.register_forward_hook(hook('upconv3'))]
        hs += [G.upconv4.register_forward_hook(hook('upconv4'))]
        if hasattr(G, 'upconv5'):
            hs += [G.upconv5.register_forward_hook(hook('upconv5'))]

        with torch.no_grad():
            final_t = G(z_t).detach().cpu()
    finally:
        for h in hs:
            h.remove()

    final = final_t.numpy()[0, 0, :].astype(np.float32)
    final_len = final.shape[0]

    order = ['z_project', 'upconv0', 'upconv1', 'upconv2', 'upconv3', 'upconv4']
    if 'upconv5' in captured:
        order.append('upconv5')

    for name in order:
        arr = captured[name][0, 0, :].numpy().astype(np.float32)
        _write_wav(arr, os.path.join(output_dir, f"{name}_raw.wav"))
        if arr.shape[0] != final_len:
            stretch = np.interp(np.linspace(0, 1, final_len, endpoint=False), np.linspace(0, 1, arr.shape[0], endpoint=False), arr)
            _write_wav(stretch.astype(np.float32), os.path.join(output_dir, f"{name}_stretched.wav"))

    _write_wav(final, os.path.join(output_dir, 'final.wav'))

    return final_len
