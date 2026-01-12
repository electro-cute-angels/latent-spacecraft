# Autore: Riccardo Petrini
import torch
from infowavegan import WaveGANGenerator as _BaseWaveGANGenerator


class WaveGANGenerator(_BaseWaveGANGenerator):
    """Variant with batchnorm enabled by default."""
    def __init__(self, *args, use_batchnorm=True, **kwargs):
        super().__init__(*args, use_batchnorm=use_batchnorm, **kwargs)
        torch.set_grad_enabled(False)
