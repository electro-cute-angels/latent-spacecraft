# Autore: Riccardo Petrini
import os
import numpy as np
import torch
import scipy.io.wavfile
from infowavegan import WaveGANGenerator

SAMPLE_RATE = 16000
SLICE_LEN = 65536


def _load_generator(ckpt_path: str, slice_len: int = SLICE_LEN):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = WaveGANGenerator(slice_len=slice_len).to(device).eval()
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    return G, device


def _write_wav(arr: np.ndarray, path: str, normalize: bool = True):
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


def generate_final(z: np.ndarray, ckpt_path: str) -> np.ndarray:
    G, device = _load_generator(ckpt_path, slice_len=SLICE_LEN)
    with torch.no_grad():
        z_t = torch.from_numpy(z).to(torch.float32).to(device)
        y = G(z_t).detach().cpu().numpy()[0, 0, :]
    return y.astype(np.float32)


def generate_evolution_pairs(z: np.ndarray, ckpt_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    G, device = _load_generator(ckpt_path, slice_len=SLICE_LEN)
    z_t = torch.from_numpy(z).to(torch.float32).to(device)

    captured = {}

    def hook(name):
        def _fn(module, inp, out):
            captured[name] = out.detach()
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

    results = []
    for name in order:
        t = captured[name][0, 0, :].cpu().numpy().astype(np.float32)
        raw_path = os.path.join(output_dir, f"{name}_raw.wav")
        stretched_path = os.path.join(output_dir, f"{name}_stretched.wav")

        _write_wav(t, raw_path, normalize=True)
        _write_wav(_resample_to_len(t, final_len), stretched_path, normalize=True)

        results.append({
            'name': name,
            'raw_path': raw_path,
            'stretched_path': stretched_path,
            'raw_len': int(t.shape[0]),
            'stretched_len': int(final_len),
        })

    final_raw_path = os.path.join(output_dir, 'final_raw.wav')
    final_stretched_path = os.path.join(output_dir, 'final_stretched.wav')
    _write_wav(final, final_raw_path, normalize=True)
    _write_wav(final, final_stretched_path, normalize=True)

    results.append({
        'name': 'final',
        'raw_path': final_raw_path,
        'stretched_path': final_stretched_path,
        'raw_len': int(final_len),
        'stretched_len': int(final_len),
    })

    return results, final_len
