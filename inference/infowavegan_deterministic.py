# Autore: Riccardo Petrini
import torch
from infowavegan import WaveGANGenerator as _BaseWaveGANGenerator


def _set_deterministic(seed: int = 42):
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class WaveGANGenerator(_BaseWaveGANGenerator):
    """Deterministic-friendly variant."""
    def __init__(self, *args, seed: int = 42, **kwargs):
        _set_deterministic(seed)
        super().__init__(*args, **kwargs)
        torch.set_grad_enabled(False)
