"""
ML Service — loads ensemble of trained models at startup, runs predictions.
Model is loaded ONCE and reused for every request (sub-200ms target).
"""
import os
import joblib
import numpy as np
from typing import Optional, Dict, Any
from ..core.config import get_settings

settings = get_settings()

# ── Module-level model cache (loaded once at startup) ─────────────────────────
_models: Dict[str, Any] = {}


def load_models():
    """
    Called once at application startup.
    Loads all available trained models into memory.
    """
    model_files = {
        "random_forest":    "ml_engine/models/rf_model.pkl",
        "gradient_boost":   "ml_engine/models/gb_model.pkl",
        "extra_trees":      "ml_engine/models/et_model.pkl",
        "ensemble":         "ml_engine/models/ensemble_model.pkl",
        "label_encoder":    "ml_engine/models/label_encoder.pkl",
        "scaler":           "ml_engine/models/scaler.pkl",
    }

    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                _models[name] = joblib.load(path)
                print(f"[ML] Loaded model: {name}")
            except Exception as e:
                print(f"[ML] Warning: Could not load {name}: {e}")
        else:
            print(f"[ML] Model not found: {path} (will skip)")


def _build_features(length: float, width: float, height: float, weight: float) -> np.ndarray:
    """
    Feature engineering — same transformations used during training.
    Features: length, width, height, weight, volume, dim_weight, aspect_ratio, density
    """
    volume      = length * width * height
    dim_weight  = volume / 5000.0
    aspect_ratio = length / max(height, 0.001)
    density     = weight / max(volume, 0.001)

    return np.array([[length, width, height, weight, volume, dim_weight, aspect_ratio, density]])


def predict_packaging(
    length: float,
    width: float,
    height: float,
    weight: float,
) -> Dict[str, Any]:
    """
    Run ML prediction using ensemble voting.
    Falls back to single model if ensemble unavailable.

    Returns:
        recommended_box, confidence_score, model_used
    """
    if length <= 0 or width <= 0 or height <= 0 or weight <= 0:
        raise ValueError("All dimensions and weight must be positive")

    features = _build_features(length, width, height, weight)

    # Scale features if scaler is available
    if "scaler" in _models:
        features = _models["scaler"].transform(features)

    # Try ensemble first (best accuracy)
    if "ensemble" in _models:
        model = _models["ensemble"]
        prediction       = model.predict(features)[0]
        probabilities    = model.predict_proba(features)[0]
        confidence_score = float(np.max(probabilities))
        model_used       = "ensemble"

    # Fallback to random forest
    elif "random_forest" in _models:
        model = _models["random_forest"]
        prediction       = model.predict(features)[0]
        probabilities    = model.predict_proba(features)[0]
        confidence_score = float(np.max(probabilities))
        model_used       = "random_forest"

    # Fallback to gradient boost
    elif "gradient_boost" in _models:
        model = _models["gradient_boost"]
        prediction       = model.predict(features)[0]
        probabilities    = model.predict_proba(features)[0]
        confidence_score = float(np.max(probabilities))
        model_used       = "gradient_boost"

    else:
        raise RuntimeError("No ML models loaded. Run ml_engine/train_models.py first.")

    # Decode label if encoder available
    if "label_encoder" in _models:
        box_type = _models["label_encoder"].inverse_transform([prediction])[0]
    else:
        box_type = str(prediction)

    return {
        "recommended_box": box_type,
        "confidence_score": round(confidence_score, 4),
        "model_used":       model_used,
    }


def get_loaded_models() -> list:
    return list(_models.keys())
