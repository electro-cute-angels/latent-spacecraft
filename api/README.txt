Author: Riccardo Petrini

What this API does
- Wraps the concatenation GAN checkpoints (TensorFlow 1-style graph) for inference.
- Provides a Cog predictor that takes a 100-D latent vector and returns a normalized WAV.

Files
- model.py: loads the frozen graph (infer.meta + model.ckpt-8956*) from weights/checkpoints/ or checkpoints/ (or GAN_CHECKPOINT_DIR).
- predict.py: Cog entrypoint; lazily loads the model and writes out.wav at the requested sample rate.
- requirements.txt: exact deps for the TF 2.9.2 runtime used by Cog.
- cog.yaml: runtime wiring for Cog.

Notes on layer outputs
- This graph exposes the final waveform tensor G_z; internal layer tensors are embedded in the graph definition (see infer.pbtxt) but not surfaced by default.
- To listen to specific internal convolutions, import the graph in debug mode and fetch tensors by name (e.g. scopes under G/). You can duplicate the approach used in GANs/inference_cached.py by registering tf.compat.v1.get_default_graph().get_tensor_by_name hooks and running a single session.run call.

Usage
- Place the checkpoint files under weights/checkpoints/ (infer/meta, infer.pbtxt, model.ckpt-8956.*) or set GAN_CHECKPOINT_DIR.
- Run via Cog: cog predict -i z:[...] -i sample_rate=16000
