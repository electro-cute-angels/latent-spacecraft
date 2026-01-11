"""TensorFlow 1.x-style loader for the concatenation GAN checkpoints.
Author: Riccardo Petrini
"""
import os
import numpy as np
import tensorflow as tf

# Force TF1 graph mode while running on TF2 runtime
_tf = tf.compat.v1
_tf.disable_v2_behavior()

__author__ = "Riccardo Petrini"


class Model:
    """Loads the frozen WaveGAN-style graph and exposes a single generate call."""

    def __init__(self, checkpoint_root: str | None = None):
        base = self._resolve_checkpoint_root(checkpoint_root)
        meta_path = os.path.join(base, "infer", "infer.meta")
        ckpt_prefix = os.path.join(base, "model.ckpt-8956")

        _tf.reset_default_graph()
        self.sess = _tf.InteractiveSession()

        saver = _tf.train.import_meta_graph(meta_path, clear_devices=True)
        saver.restore(self.sess, ckpt_prefix)

        graph = _tf.get_default_graph()
        self.input = graph.get_tensor_by_name("z:0")
        self.output = graph.get_tensor_by_name("G_z:0")[:, :, 0]

    def _resolve_checkpoint_root(self, override: str | None) -> str:
        if override and os.path.isdir(override):
            return override
        cwd = os.getcwd()
        candidates = [
            os.path.join(cwd, "weights", "checkpoints"),
            os.path.join(cwd, "checkpoints"),
            os.environ.get("GAN_CHECKPOINT_DIR", ""),
        ]
        for path in candidates:
            if path and os.path.isdir(path):
                return path
        raise FileNotFoundError(
            "No checkpoints found. Set GAN_CHECKPOINT_DIR or place weights/checkpoints here."
        )

    def generate_audio(self, z_array: np.ndarray) -> np.ndarray:
        feed = {self.input: z_array}
        output = self.sess.run(self.output, feed_dict=feed)
        audio = output[0]
        peak = np.max(np.abs(audio)) or 1.0
        return (audio / peak).astype(np.float32)


def load_model() -> Model:
    return Model()
