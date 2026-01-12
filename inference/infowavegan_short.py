# Autore: Riccardo Petrini
import torch
from infowavegan import WaveGANGenerator as _BaseWaveGANGenerator


class WaveGANGenerator(_BaseWaveGANGenerator):
    """Short-slice variant (16384) with default width."""
    def __init__(self, *args, slice_len=16384, **kwargs):
        super().__init__(*args, slice_len=slice_len, **kwargs)
        torch.set_grad_enabled(False)
