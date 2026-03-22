"""
ML Service — loads pre-trained models at startup.
If model files are missing (e.g. fresh Render deploy), auto-trains on
synthetic data so ML is always available without committing large .pkl files.

Decision hierarchy:
  1. Rule-based FFD (handles 95%+ of orders)
  2. ML fallback (only when FFD finds no valid box)
"""
import os
import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

_models: Dict[str, Any] = {}
_sklearn_available = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    _sklearn_available = True
    logger.info("[ML] scikit-learn available")
except ImportError:
    logger.warning("[ML] scikit-learn not installed — rule engine handles all orders")


# ── Synthetic training data — mirrors Indian ecommerce box catalogue ──────────
BOX_CONFIGS = [
    ("Box_XS",  (5,15),  (5,10),  (5,10),  (0.05,0.5)),
    ("Box_S",   (15,25), (10,20), (5,15),  (0.3, 3.0)),
    ("Box_M",   (25,40), (15,30), (10,25), (1.0, 8.0)),
    ("Box_L",   (35,55), (25,45), (20,35), (3.0,15.0)),
    ("Box_XL",  (50,75), (35,55), (30,50), (8.0,25.0)),
    ("Box_XXL", (70,100),(50,80), (50,70), (15.0,40.0)),
]

FEATURE_COLS = ["length","width","height","weight",
                "volume","dim_weight","aspect_ratio","density","lwh_sum","lw_ratio"]


def _generate_data(n=3000):
    import random
    rng   = np.random.RandomState(42)
    rows  = []
    per   = n // len(BOX_CONFIGS)
    for box_type, lr, wr, hr, wtr in BOX_CONFIGS:
        for _ in range(per):
            l = float(np.clip(rng.uniform(*lr) + rng.normal(0,1.5), 1, 200))
            w = float(np.clip(rng.uniform(*wr) + rng.normal(0,1.2), 1, 200))
            h = float(np.clip(rng.uniform(*hr) + rng.normal(0,1.0), 1, 200))
            wt= float(np.clip(rng.uniform(*wtr)+ rng.normal(0,0.3), 0.05,100))
            rows.append((round(l,2), round(w,2), round(h,2), round(wt,3), box_type))
    rng.shuffle(rows)
    return rows


def _build_X(rows):
    X = []
    for l, w, h, wt, _ in rows:
        vol = l*w*h
        X.append([l, w, h, wt,
                  vol, vol/5000,
                  l/max(h,0.001), wt/max(vol,0.001),
                  l+w+h, l/max(w,0.001)])
    return np.array(X)


def _train():
    logger.info("[ML] Training models on synthetic data (n=3000)...")
    rows   = _generate_data(3000)
    X_raw  = _build_X(rows)
    labels = [r[4] for r in rows]

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    sc = StandardScaler()
    X  = sc.fit_transform(X_raw)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    ensemble = VotingClassifier(
        estimators=[("rf", RandomForestClassifier(n_estimators=80,n_jobs=-1,random_state=42)),
                    ("gb", GradientBoostingClassifier(n_estimators=80,learning_rate=0.1,random_state=42))],
        voting="soft", n_jobs=-1,
    )

    rf.fit(X, y)
    gb.fit(X, y)
    ensemble.fit(X, y)

    # Quick accuracy check
    preds  = ensemble.predict(X)
    acc    = float(np.mean(preds == y))
    logger.info(f"[ML] Training complete — ensemble accuracy: {acc:.2%}")

    _models["random_forest"]  = rf
    _models["gradient_boost"] = gb
    _models["ensemble"]       = ensemble
    _models["label_encoder"]  = le
    _models["scaler"]         = sc
    logger.info(f"[ML] 5 models ready in memory")


def load_models():
    """
    Called once at startup.
    Tries to load from disk first.
    If files missing, auto-trains in memory (~8 seconds).
    """
    if not _sklearn_available:
        logger.info("[ML] sklearn not available — skipping")
        return

    import joblib
    model_dir = os.path.join(os.path.dirname(__file__),
                             "../../../../ml_engine/models")
    model_dir = os.path.normpath(model_dir)

    files = {
        "ensemble":      os.path.join(model_dir, "ensemble_model.pkl"),
        "random_forest": os.path.join(model_dir, "rf_model.pkl"),
        "label_encoder": os.path.join(model_dir, "label_encoder.pkl"),
        "scaler":        os.path.join(model_dir, "scaler.pkl"),
    }

    # Try loading from disk
    loaded = 0
    for name, path in files.items():
        if os.path.exists(path):
            try:
                _models[name] = joblib.load(path)
                loaded += 1
            except Exception as e:
                logger.warning(f"[ML] Could not load {name}: {e}")

    if loaded >= 3:
        logger.info(f"[ML] Loaded {loaded} models from disk")
        return

    # Files missing — train in memory
    logger.info("[ML] Model files not found — auto-training in memory")
    try:
        _train()
    except Exception as e:
        logger.error(f"[ML] Auto-train failed: {e}")


def _validate_inputs(l: float, w: float, h: float, wt: float):
    errors = []
    if not (0.1 <= l  <= 200): errors.append(f"length {l} out of range")
    if not (0.1 <= w  <= 200): errors.append(f"width {w} out of range")
    if not (0.1 <= h  <= 200): errors.append(f"height {h} out of range")
    if not (0.01<= wt <= 100): errors.append(f"weight {wt} out of range")
    if errors:
        raise ValueError(f"Invalid inputs: {'; '.join(errors)}")


def _build_features(l, w, h, wt):
    vol = l*w*h
    return np.array([[l, w, h, wt,
                       vol, vol/5000,
                       l/max(h,0.001), wt/max(vol,0.001),
                       l+w+h, l/max(w,0.001)]])


def predict_packaging(l: float, w: float, h: float, wt: float) -> Dict[str, Any]:
    if not _sklearn_available:
        raise RuntimeError("scikit-learn not installed")
    if not _models:
        raise RuntimeError("No models loaded")

    _validate_inputs(l, w, h, wt)

    feat = _build_features(l, w, h, wt)
    if "scaler" in _models:
        feat = _models["scaler"].transform(feat)

    for name in ["ensemble", "random_forest", "gradient_boost"]:
        if name not in _models:
            continue
        pred  = _models[name].predict(feat)[0]
        proba = _models[name].predict_proba(feat)[0]
        conf  = float(np.max(proba))
        box   = _models["label_encoder"].inverse_transform([pred])[0] \
                if "label_encoder" in _models else str(pred)

        logger.info(f"[ML] {box} conf={conf:.2%} model={name} dims={l}x{w}x{h}cm wt={wt}kg")
        return {"recommended_box": box, "confidence_score": round(conf,4), "model_used": name}

    raise RuntimeError("No usable model")


def get_loaded_models() -> List[str]:
    return list(_models.keys())


def is_ml_available() -> bool:
    return _sklearn_available and len(_models) >= 3
