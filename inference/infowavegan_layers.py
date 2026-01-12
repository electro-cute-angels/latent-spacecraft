# Autore: Riccardo Petrini
import torch
from infowavegan import WaveGANGenerator as _BaseWaveGANGenerator


class WaveGANGenerator(_BaseWaveGANGenerator):
    """Variant that exposes a helper to get hookable modules."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.set_grad_enabled(False)

    def layer_map(self):
        return {
            'z_project': self.z_batchnorm,
            'upconv0': self.upconv0,
            'upconv1': self.upconv1,
            'upconv2': self.upconv2,
            'upconv3': self.upconv3,
            'upconv4': self.upconv4,
            'upconv5': getattr(self, 'upconv5', None),
        }
