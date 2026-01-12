Author: Riccardo Petrini

Variations (pair each inference_*.py with its infowavegan_*):
- inference_base.py / infowavegan_base.py: baseline hooks + batchnorm-on generator, writes raw/stretched layers and final.
- inference_fastgpu.py / infowavegan_fastgpu.py: CUDA-first, autocast/FP16, wider channels for richer spectra.
- inference_cpusafe.py / infowavegan_cpusafe.py: CPU-only, narrower channels for low-RAM boxes.
- inference_short.py / infowavegan_short.py: short-slice (16384) runs for quick previews or lightweight checkpoints.
- inference_layers.py / infowavegan_layers.py: returns a tensor dict (no disk writes) for downstream processing; exposes layer map helper.
- inference_deterministic.py / infowavegan_deterministic.py: seeds and deterministic algorithms for reproducible passes; writes layer audio.

!!!!!!!!!!!! ricorda di 16kHz e 16384 slices (guarda checkpoints tho..)
