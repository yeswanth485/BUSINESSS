"""
ML Service — loads pre-trained models at startup if available.
If scikit-learn is not installed (e.g. on Render free tier),
the service gracefully disables ML and the decision engine
falls back to the rule-based system automatically.
"""
import os
import joblib
import numpy as np
from typing import Optional, Dict, Any, List

_models: Dict[str, Any] = {}
_sklearn_available = False

# Try to import sklearn — optional on server
try:
    import sklearn  # noqa
    _sklearn_available = True
    print("[ML] scikit-learn available")
except ImportError:
    print("[ML] scikit-learn not installed — ML fallback disabled, rule-based engine will handle all orders")


def load_models():
    """
    Called once at startup.
    Only loads models if sklearn is available and .pkl files exist.
    """
    if not _sklearn_available:
        print("[ML] Skipping model load — sklearn not available")
        return

    model_files = {
        "random_forest":  "ml_engine/models/rf_model.pkl",
        "gradient_boost": "ml_engine/models/gb_model.pkl",
        "extra_trees":    "ml_engine/models/et_model.pkl",
        "ensemble":       "ml_engine/models/ensemble_model.pkl",
        "label_encoder":  "ml_engine/models/label_encoder.pkl",
        "scaler":         "ml_engine/models/scaler.pkl",
    }

    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                _models[name] = joblib.load(path)
                print(f"[ML] Loaded: {name}")
            except Exception as e:
                print(f"[ML] Could not load {name}: {e}")
        else:
            print(f"[ML] Not found: {path}")


def _build_features(length: float, width: float, height: float, weight: float):
    volume       = length * width * height
    dim_weight   = volume / 5000.0
    aspect_ratio = length / max(height, 0.001)
    density      = weight / max(volume, 0.001)
    lwh_sum      = length + width + height
    lw_ratio     = length / max(width, 0.001)
    return np.array([[length, width, height, weight,
                       volume, dim_weight, aspect_ratio,
                       density, lwh_sum, lw_ratio]])


def predict_packaging(
    length: float,
    width: float,
    height: float,
    weight: float,
) -> Dict[str, Any]:
    """
    Run ML prediction. Raises RuntimeError if models unavailable
    so decision_service can catch it and use rule-based fallback.
    """
    if not _sklearn_available:
        raise RuntimeError("scikit-learn not installed on this server")

    if not _models:
        raise RuntimeError("No ML models loaded")

    if any(v <= 0 for v in [length, width, height, weight]):
        raise ValueError("All dimensions and weight must be positive")

    features = _build_features(length, width, height, weight)

    if "scaler" in _models:
        features = _models["scaler"].transform(features)

    # Try best model first, fall through
    for model_name in ["ensemble", "random_forest", "gradient_boost", "extra_trees"]:
        if model_name in _models:
            model         = _models[model_name]
            prediction    = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence    = float(np.max(probabilities))

            if "label_encoder" in _models:
                box_type = _models["label_encoder"].inverse_transform([prediction])[0]
            else:
                box_type = str(prediction)

            return {
                "recommended_box":  box_type,
                "confidence_score": round(confidence, 4),
                "model_used":       model_name,
            }

    raise RuntimeError("No usable ML model found")


def get_loaded_models() -> List[str]:
    return list(_models.keys())


def is_ml_available() -> bool:
    return _sklearn_available and len(_models) > 0
