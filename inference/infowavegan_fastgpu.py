# Autore: Riccardo Petrini
import torch
from infowavegan import WaveGANGenerator as _BaseWaveGANGenerator


class WaveGANGenerator(_BaseWaveGANGenerator):
    """GPU-leaning variant with larger channels and batchnorm."""
    def __init__(self, *args, dim=96, use_batchnorm=True, **kwargs):
        super().__init__(*args, dim=dim, use_batchnorm=use_batchnorm, **kwargs)
        torch.set_grad_enabled(False)
