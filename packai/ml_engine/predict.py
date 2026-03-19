"""
Prediction Module — standalone predict function for testing.
The backend uses ml_service.py which caches models in memory.
"""
import os
import joblib
import numpy as np
from typing import Dict, Any


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def _load(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}. Run train_models.py first.")
    return joblib.load(path)


def predict_packaging(
    length: float,
    width: float,
    height: float,
    weight: float,
    model_name: str = "ensemble",
) -> Dict[str, Any]:
    """
    Predict best box type for given product dimensions.

    Args:
        length, width, height: product dimensions in cm
        weight:                product weight in kg
        model_name:            ensemble | random_forest | gradient_boost | extra_trees | svm

    Returns:
        dict with recommended_box, confidence_score, model_used
    """
    if any(v <= 0 for v in [length, width, height, weight]):
        raise ValueError("All dimensions and weight must be positive numbers.")

    # Feature engineering — must match training
    volume       = length * width * height
    dim_weight   = volume / 5000.0
    aspect_ratio = length / max(height, 0.001)
    density      = weight / max(volume, 0.001)
    lwh_sum      = length + width + height
    lw_ratio     = length / max(width, 0.001)

    features = np.array([[
        length, width, height, weight,
        volume, dim_weight, aspect_ratio, density, lwh_sum, lw_ratio
    ]])

    # Scale
    scaler   = _load("scaler.pkl")
    features = scaler.transform(features)

    # Load model
    model_files = {
        "ensemble":       "ensemble_model.pkl",
        "random_forest":  "rf_model.pkl",
        "gradient_boost": "gb_model.pkl",
        "extra_trees":    "et_model.pkl",
        "svm":            "svm_model.pkl",
    }
    model_file = model_files.get(model_name, "ensemble_model.pkl")
    model      = _load(model_file)
    le         = _load("label_encoder.pkl")

    prediction    = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence    = float(np.max(probabilities))
    box_type      = le.inverse_transform([prediction])[0]

    return {
        "recommended_box":  box_type,
        "confidence_score": round(confidence, 4),
        "model_used":       model_name,
    }


if __name__ == "__main__":
    # Quick smoke test
    test_cases = [
        (8,  6,  5,  0.2,  "Small accessory"),
        (20, 15, 10, 1.5,  "Book / small electronics"),
        (35, 25, 20, 4.0,  "Clothing / medium item"),
        (55, 40, 30, 12.0, "Shoes / large gadget"),
        (80, 60, 50, 22.0, "Large appliance"),
    ]
    print("\nPackAI ML Prediction Test\n" + "-"*50)
    for l, w, h, wt, label in test_cases:
        result = predict_packaging(l, w, h, wt)
        print(f"{label:<25} → {result['recommended_box']}  (conf: {result['confidence_score']:.2%})")
