# Autore: Riccardo Petrini
import torch
import numpy as np
from infowavegan_layers import WaveGANGenerator

SAMPLE_RATE = 16000
SLICE_LEN = 65536


def _load_generator(ckpt_path: str, slice_len: int = SLICE_LEN):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = WaveGANGenerator(slice_len=slice_len).to(device).eval()
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    return G, device


def run_with_hooks(z: np.ndarray, ckpt_path: str):
    G, device = _load_generator(ckpt_path, slice_len=SLICE_LEN)
    z_t = torch.from_numpy(z).to(torch.float32).to(device)
    layers = {}

    def hook(name):
        def _fn(module, inp, out):
            layers[name] = out.detach().cpu()
        return _fn

    hs = []
    lm = G.layer_map()
    try:
        for name, module in lm.items():
            if module is None:
                continue
            hs.append(module.register_forward_hook(hook(name)))
        with torch.no_grad():
            final_t = G(z_t).detach().cpu()
    finally:
        for h in hs:
            h.remove()

    layers['final'] = final_t
    return layers


def to_numpy(layer_tensor):
    if hasattr(layer_tensor, 'cpu'):
        return layer_tensor[0, 0, :].cpu().numpy().astype(np.float32)
    return np.asarray(layer_tensor, dtype=np.float32)
