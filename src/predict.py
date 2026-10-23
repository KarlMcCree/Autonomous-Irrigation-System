# src/predict.py

import json
import numpy as np

from . import config


# ═══════════════════════════════════════════════════════════════
# SOIL MOISTURE PREDICTOR (LSTM)
# ═══════════════════════════════════════════════════════════════
class SoilPredictor:

    def __init__(self, model, meta: dict):
        self.model = model
        self.meta = meta
        self.window_size = meta.get("window_size", config.WINDOW_SIZE)
        self.n_features = meta.get("n_features", config.N_FEATURES)
        self.horizon = meta.get("horizon", config.PREDICT_HORIZON)
        self.feature_names = meta.get("feature_names",
                                       ["soil_moisture", "temperature", "humidity"])

    @classmethod
    def load(cls, model_path=config.SOIL_MODEL_KERAS,
             meta_path=config.SOIL_MODEL_META):
        """Load model + metadata from disk. Verifies compatibility."""
        import os
        import tensorflow as tf

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Soil model not found: {model_path}\n"
                f"Train it first: python -m src.train_soil"
            )

        print(f" Loading soil predictor: {model_path}")
        model = tf.keras.models.load_model(model_path)

        # Load metadata if it exists (graceful fallback if not)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            print(f"  No metadata at {meta_path} — using config defaults")
            meta = {
                "window_size": config.WINDOW_SIZE,
                "n_features": config.N_FEATURES,
                "horizon": config.PREDICT_HORIZON,
            }

        # Verify shape compatibility
        expected_shape = (None, meta["window_size"], meta["n_features"])
        actual_shape = model.input_shape
        if actual_shape[1:] != expected_shape[1:]:
            raise ValueError(
                f" Model expects input shape {actual_shape}, "
                f"but metadata says {expected_shape}"
            )

        print(f" Soil predictor ready: window={meta['window_size']}, "
              f"horizon={meta['horizon']}min")
        return cls(model, meta)

    def predict(self, history) -> float:
        """
        Predict soil moisture `horizon` minutes into the future.

        Args:
            history: sequence of [soil, temp, humid] rows,
                     length must be >= window_size.

        Returns:
            Predicted soil moisture in [0, 1].
        """
        if len(history) < self.window_size:
            raise ValueError(
                f"Need at least {self.window_size} samples, "
                f"got {len(history)}"
            )

        # Take the LAST window_size samples
        window = np.array(history[-self.window_size:], dtype="float32")
        x = window[None, :, :]  # (1, W, F)
        y = self.model.predict(x, verbose=0)
        return float(np.clip(y[0, 0], 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════
# PLANT DISEASE CLASSIFIER (from Project 2)
# ═══════════════════════════════════════════════════════════════
class PlantClassifier:
    """
    Wraps the tomato disease classifier from Project 2.

    Copy the trained model + meta from your Project 2 folder:
        ml/plant_classifier.keras
        ml/plant_meta.json
    """

    def __init__(self, model, meta: dict):
        self.model = model
        self.meta = meta
        self.classes = meta["classes"]
        self.display = meta.get("class_display", {c: c for c in self.classes})
        self.img_size = meta.get("img_size", 96)
        self.uses_preprocess_input = meta.get("uses_preprocess_input", False)

    @classmethod
    def load(cls, model_path=config.PLANT_MODEL_KERAS,
             meta_path=config.PLANT_MODEL_META):
        import os
        import tensorflow as tf

        if not os.path.exists(model_path):
            print(f"  Plant model not found: {model_path}")
            print(f"  Vision-based decisions will be skipped.")
            return None

        print(f" Loading plant classifier: {model_path}")
        model = tf.keras.models.load_model(model_path)
        with open(meta_path) as f:
            meta = json.load(f)

        return cls(model, meta)

    def _preprocess(self, img_rgb):
        """Match the preprocessing used during Project 2 training."""
        import cv2
        img = cv2.resize(img_rgb, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_AREA)
        if self.uses_preprocess_input:
            return img.astype("float32")           # keep [0, 255]
        return img.astype("float32") / 255.0       # normalize [0, 1]

    def predict_array(self, img_rgb) -> dict:
        """Predict from an RGB numpy array."""
        x = np.expand_dims(self._preprocess(img_rgb), axis=0)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return {
            "class": self.classes[idx],
            "display_name": self.display[self.classes[idx]],
            "confidence": float(probs[idx]),
            "all_probs": {
                self.display[c]: float(p)
                for c, p in zip(self.classes, probs)
            },
        }