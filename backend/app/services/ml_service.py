"""
ML Service — loads pre-trained models at startup.
Validates inputs strictly before prediction.
Logs all predictions for monitoring.
Only used as fallback when rule-based engine fails.
"""
import os
import logging
import numpy as np
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

_models: Dict[str, Any] = {}
_sklearn_available = False

try:
    import sklearn
    _sklearn_available = True
    logger.info("[ML] scikit-learn available")
except ImportError:
    logger.warning("[ML] scikit-learn not installed — rule-based engine handles all orders")


def load_models():
    if not _sklearn_available:
        logger.info("[ML] Skipping model load — sklearn not available")
        return

    import joblib
    model_files = {
        "ensemble":      "ml_engine/models/ensemble_model.pkl",
        "random_forest": "ml_engine/models/rf_model.pkl",
        "gradient_boost":"ml_engine/models/gb_model.pkl",
        "extra_trees":   "ml_engine/models/et_model.pkl",
        "label_encoder": "ml_engine/models/label_encoder.pkl",
        "scaler":        "ml_engine/models/scaler.pkl",
    }
    loaded = 0
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                _models[name] = joblib.load(path)
                loaded += 1
                logger.info(f"[ML] Loaded: {name}")
            except Exception as e:
                logger.error(f"[ML] Failed to load {name}: {e}")
        else:
            logger.debug(f"[ML] Not found: {path}")
    logger.info(f"[ML] {loaded}/{len(model_files)} models loaded")


def _validate_inputs(length: float, width: float, height: float, weight: float):
    """Strict input validation before ML prediction."""
    errors = []
    if not (0.1 <= length <= 200):  errors.append(f"length {length} out of range [0.1, 200]")
    if not (0.1 <= width  <= 200):  errors.append(f"width {width} out of range [0.1, 200]")
    if not (0.1 <= height <= 200):  errors.append(f"height {height} out of range [0.1, 200]")
    if not (0.01 <= weight <= 100): errors.append(f"weight {weight} out of range [0.01, 100]")
    if errors:
        raise ValueError(f"Invalid inputs: {'; '.join(errors)}")


def _build_features(length: float, width: float, height: float, weight: float) -> np.ndarray:
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
    length: float, width: float, height: float, weight: float
) -> Dict[str, Any]:
    """
    ML prediction with strict validation and logging.
    Raises RuntimeError if unavailable — caller uses rule-based fallback.
    """
    if not _sklearn_available:
        raise RuntimeError("scikit-learn not installed on this server")
    if not _models:
        raise RuntimeError("No ML models loaded — using rule-based engine")

    _validate_inputs(length, width, height, weight)

    features = _build_features(length, width, height, weight)
    if "scaler" in _models:
        features = _models["scaler"].transform(features)

    for model_name in ["ensemble", "random_forest", "gradient_boost", "extra_trees"]:
        if model_name in _models:
            model         = _models[model_name]
            prediction    = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence    = float(np.max(probabilities))

            box_type = (
                _models["label_encoder"].inverse_transform([prediction])[0]
                if "label_encoder" in _models
                else str(prediction)
            )

            logger.info(f"[ML] Predicted {box_type} (conf={confidence:.2%}, model={model_name}, dims={length}x{width}x{height}cm, wt={weight}kg)")

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
