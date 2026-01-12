# Autore: Riccardo Petrini
import os
import numpy as np
import torch
import scipy.io.wavfile
from infowavegan_fastgpu import WaveGANGenerator

SAMPLE_RATE = 16000
SLICE_LEN = 65536


def _load_generator(ckpt_path: str, slice_len: int = SLICE_LEN):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = WaveGANGenerator(slice_len=slice_len).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(state)
    return G, device


def _write_wav(arr: np.ndarray, path: str, normalize: bool = True):
    x = arr.astype(np.float32)
    if normalize:
        m = np.max(np.abs(x))
        if m > 0:
            x = x / m
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scipy.io.wavfile.write(path, SAMPLE_RATE, x)


def generate_final(z: np.ndarray, ckpt_path: str) -> np.ndarray:
    G, device = _load_generator(ckpt_path, slice_len=SLICE_LEN)
    z_t = torch.from_numpy(z).to(torch.float16 if device.type == 'cuda' else torch.float32).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            y = G(z_t).detach().cpu().numpy()[0, 0, :]
    return y.astype(np.float32)


def generate_layers(z: np.ndarray, ckpt_path: str):
    G, device = _load_generator(ckpt_path, slice_len=SLICE_LEN)
    z_t = torch.from_numpy(z).to(torch.float16 if device.type == 'cuda' else torch.float32).to(device)
    captured = {}

    def hook(name):
        def _fn(module, inp, out):
            captured[name] = out.detach().float()
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
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                final_t = G(z_t).detach().cpu()
    finally:
        for h in hs:
            h.remove()

    final = final_t.numpy()[0, 0, :].astype(np.float32)
    captured['final'] = final_t
    return captured, final


def dump_layers(captured: dict, final: np.ndarray, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    final_len = final.shape[0]
    for name, tensor in captured.items():
        arr = tensor[0, 0, :].cpu().numpy().astype(np.float32) if hasattr(tensor, 'cpu') else tensor
        raw_path = os.path.join(output_dir, f"{name}_raw.wav")
        _write_wav(arr, raw_path, normalize=True)
        if arr.shape[0] != final_len:
            stretch = np.interp(np.linspace(0, 1, final_len, endpoint=False), np.linspace(0, 1, arr.shape[0], endpoint=False), arr)
            _write_wav(stretch.astype(np.float32), os.path.join(output_dir, f"{name}_stretched.wav"), normalize=True)
    _write_wav(final, os.path.join(output_dir, 'final.wav'), normalize=True)

