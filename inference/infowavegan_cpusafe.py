# Autore: Riccardo Petrini
import torch
from infowavegan import WaveGANGenerator as _BaseWaveGANGenerator


class WaveGANGenerator(_BaseWaveGANGenerator):
    """CPU-safe variant with smaller channel width."""
    def __init__(self, *args, dim=48, use_batchnorm=False, **kwargs):
        super().__init__(*args, dim=dim, use_batchnorm=use_batchnorm, **kwargs)
        torch.set_grad_enabled(False)
