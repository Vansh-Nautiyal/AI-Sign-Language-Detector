"""
predict.py — Real-Time ASL Prediction Engine
=============================================
Core prediction logic: load model, process landmarks, apply smoothing.
Imported by app.py and usable standalone.

All stateless helpers (normalization, feature extraction, voting,
label encoding) are delegated to utils.py.
"""

import os
import numpy as np
import tensorflow as tf

from utils import extract_features, MajorityVoter, ASLLabelEncoder   # ← utils.py

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "asl_model.h5")
LABEL_PATH  = os.path.join(BASE_DIR, "model", "class_names.txt")

# ── Thresholds ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60   # only show prediction if model is confident enough
SMOOTHING_WINDOW     = 10     # majority-vote over last N predictions


class ASLPredictor:
    """Wraps the Keras model with normalization, smoothing, and thresholding."""

    def __init__(
        self,
        model_path: str   = MODEL_PATH,
        label_path: str   = LABEL_PATH,
        confidence: float = CONFIDENCE_THRESHOLD,
        window: int       = SMOOTHING_WINDOW,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model not found at:\n  {model_path}\n"
                "Run  src/train_model.py  first."
            )

        self.model      = tf.keras.models.load_model(model_path)
        self.encoder    = ASLLabelEncoder.load(label_path)   # utils label encoder
        self.confidence = confidence
        self._voter     = MajorityVoter(window=window)       # utils majority voter

    @property
    def class_names(self) -> list[str]:
        """Convenience passthrough to the encoder's class list."""
        return self.encoder.class_names

    # ── Single-frame prediction ───────────────────────────────────────────────
    def predict_single(self, landmarks):
        """
        Run one inference pass.

        Parameters
        ----------
        landmarks : sequence of MediaPipe NormalizedLandmark objects

        Returns
        -------
        letter     : str    — predicted letter
        confidence : float  — [0, 1]
        raw_probs  : ndarray shape (num_classes,) — full softmax output
        """
        features  = extract_features(landmarks)          # utils feature extraction
        raw_probs = self.model.predict(features, verbose=0)[0]
        idx       = int(np.argmax(raw_probs))
        conf      = float(raw_probs[idx])
        letter    = self.encoder.decode(idx)             # utils label decoding
        return letter, conf, raw_probs

    # ── Smoothed prediction ───────────────────────────────────────────────────
    def predict(self, landmarks):
        """
        Run inference with confidence thresholding and majority-vote smoothing.

        Returns
        -------
        letter   : str | None — raw prediction (None if below threshold)
        confidence : float
        smoothed : str | None — majority-voted letter across recent frames
        """
        letter, conf, _ = self.predict_single(landmarks)

        if conf < self.confidence:
            # Below threshold: do not keep showing a stale previous letter.
            return None, conf, None

        self._voter.update(letter)                       # utils MajorityVoter
        return letter, conf, self._voter.vote()

    def predict_with_probabilities(self, landmarks):
        """
        Run inference with smoothing and include the full softmax output.

        Returns
        -------
        letter : str | None
        confidence : float
        smoothed : str | None
        raw_probs : ndarray
        """
        letter, conf, raw_probs = self.predict_single(landmarks)

        if conf < self.confidence:
            return None, conf, None, raw_probs

        self._voter.update(letter)
        return letter, conf, self._voter.vote(), raw_probs

    def reset(self) -> None:
        """Clear the smoothing buffer (e.g., when the hand leaves frame)."""
        self._voter.reset()                              # utils MajorityVoter