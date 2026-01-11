Author: Riccardo Petrini

What lives here
- inference_cached.py: lazy WaveGAN loader with cached generator; produces audio and captures intermediate layers via forward hooks.
- server_fastapi.py: FastAPI service wrapping the generator and (optional) Whisper ASR. Routes: /generate, /generate_file, /generate_evolution, /generate_layer_file, /transcribe_file.
- server_flask_min.py: Minimal Flask service with the same GAN backend. Routes: /generate, /generate_file, /layers.

How the GAN outputs are produced
- The generator is loaded once (WaveGANGenerator) and runs on CPU or CUDA depending on availability.
- A 100-D latent vector z in [-1,1] is sampled; the first 16 slots can be overwritten by the binary categorical_code payload.
- The generator forward produces a mono waveform (float32) of length 65536 at 16 kHz. Audio is normalized and written as WAV.

How to extract internal convolutional layers
- inference_cached.WaveGANService.generate_layers registers forward hooks on z_project, upconv0..upconv4 (and upconv5 if present).
- Each hook captures the layer output tensor during a single forward pass.
- For every captured layer, two WAVs are written:
  * <name>_raw.wav: the layer activation as-is (channel 0) with its native length.
  * <name>_stretched.wav: the same activation resampled to the final audio length so you can listen to it.
- The method returns a list of dicts describing paths and lengths; the final waveform is included as the last item.

Endpoints for layer outputs
- Flask (/layers): POST {"categorical_code": [16 binary ints]} -> returns URLs for raw/stretched WAVs for each layer.
- FastAPI (/generate_evolution): POST same body -> returns layer metadata and URLs for raw/stretched WAVs.
- FastAPI (/generate_layer_file): POST {"categorical_code": [...], "layer": "upconv3"} -> copies the stretched layer output to a unique WAV in static/batch and returns its URL.

Running
- Set GAN_CHECKPOINT to your generator checkpoint path (defaults to checkpoint/epoch450_step166500_G.pt).
- FastAPI: uvicorn GANs.server_fastapi:app --host 0.0.0.0 --port 8000
- Flask: python GANs/server_flask_min.py --port 5000
