"""YAMNet TFLite Gunshot Classifier simulation matching Android detector."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "app", "src", "main", "assets", "yamnet.tflite"
)


@dataclass
class ClassificationResult:
    gunshot_confidence: float
    top_label: str
    top_score: float
    best_gunshot_label: str
    best_gunshot_index: int
    scores: np.ndarray


class YamnetClassifier:
    INPUT_SAMPLES = 15600  # ~0.975 seconds at 16kHz

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        class_map=None,
        mode: str = "tightened",
    ):
        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        from eval.class_map import YamnetClassMap
        self.class_map = class_map or YamnetClassMap()
        self.mode = mode
        self.gunshot_indices = self.class_map.get_indices(mode)

        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def set_mode(self, mode: str):
        self.mode = mode
        self.gunshot_indices = self.class_map.get_indices(mode)

    def classify(self, audio_16k: np.ndarray) -> Optional[ClassificationResult]:
        """
        Run inference on 16kHz mono audio.
        Audio should be normalized to [-1.0, 1.0].
        If length != 15600, it is zero-padded or truncated to match input shape.
        """
        # Ensure 1D float32 array
        audio = np.asarray(audio_16k, dtype=np.float32).flatten()
        if len(audio) < self.INPUT_SAMPLES:
            padded = np.zeros(self.INPUT_SAMPLES, dtype=np.float32)
            padded[: len(audio)] = audio
            audio = padded
        elif len(audio) > self.INPUT_SAMPLES:
            audio = audio[-self.INPUT_SAMPLES :]

        # Bundled yamnet.tflite expects 1D input shape [15600]
        self.interpreter.set_tensor(self.input_details[0]["index"], audio)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Output shape is [1, 521]
        scores = output[0]
        top_idx = int(np.argmax(scores))
        top_score = float(scores[top_idx])
        top_label = self.class_map.get_label(top_idx)

        gunshot_score = 0.0
        best_gunshot_idx = -1
        for idx in self.gunshot_indices:
            if idx < len(scores) and scores[idx] > gunshot_score:
                gunshot_score = float(scores[idx])
                best_gunshot_idx = idx

        best_label = (
            self.class_map.get_label(best_gunshot_idx) if best_gunshot_idx >= 0 else "None"
        )

        return ClassificationResult(
            gunshot_confidence=gunshot_score,
            top_label=top_label,
            top_score=top_score,
            best_gunshot_label=best_label,
            best_gunshot_index=best_gunshot_idx,
            scores=scores,
        )
