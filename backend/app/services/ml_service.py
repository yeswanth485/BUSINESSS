"""
ML Service — 5 models, all run on every order, majority vote decides.

Models:
  1. Random Forest       — fast, handles noise well
  2. Gradient Boosting   — high accuracy on structured data
  3. Extra Trees         — fast, diverse from RF
  4. SVM (RBF kernel)    — strong on small dimensional spaces
  5. Voting Ensemble     — combines RF+GB+ET with soft voting

Flow per order:
  - All 5 models predict independently
  - Each gives a box recommendation + confidence score
  - Majority vote (weighted by confidence) selects final box
  - FFD then validates the voted box physically fits
"""
import os
import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

_models: Dict[str, Any] = {}
_sklearn_available = False

try:
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        ExtraTreesClassifier,
        VotingClassifier,
    )
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    _sklearn_available = True
    logger.info("[ML] scikit-learn available — 5-model ensemble ready")
except ImportError:
    logger.warning("[ML] scikit-learn not installed — FFD rule engine handles all orders")


# ── Box catalogue — Indian ecommerce dimensions ───────────────────────────────
BOX_CONFIGS = [
    ("Box_XS",  (5,15),  (5,10),  (5,10),  (0.05, 0.5)),
    ("Box_S",   (15,25), (10,20), (5,15),  (0.3,  3.0)),
    ("Box_M",   (25,40), (15,30), (10,25), (1.0,  8.0)),
    ("Box_L",   (35,55), (25,45), (20,35), (3.0,  15.0)),
    ("Box_XL",  (50,75), (35,55), (30,50), (8.0,  25.0)),
    ("Box_XXL", (70,100),(50,80), (50,70), (15.0, 40.0)),
]


def _generate_data(n: int = 4800):
    """Generate realistic synthetic Indian ecommerce data — 800 samples per box."""
    rng  = np.random.RandomState(42)
    rows = []
    per  = n // len(BOX_CONFIGS)
    for box_type, lr, wr, hr, wtr in BOX_CONFIGS:
        for _ in range(per):
            l  = float(np.clip(rng.uniform(*lr)  + rng.normal(0, 1.5), 1, 200))
            w  = float(np.clip(rng.uniform(*wr)  + rng.normal(0, 1.2), 1, 200))
            h  = float(np.clip(rng.uniform(*hr)  + rng.normal(0, 1.0), 1, 200))
            wt = float(np.clip(rng.uniform(*wtr) + rng.normal(0, 0.3), 0.05, 100))
            rows.append((round(l,2), round(w,2), round(h,2), round(wt,3), box_type))
    rng.shuffle(rows)
    return rows


def _build_features_batch(rows):
    """Build 10-feature vectors for a list of (l,w,h,wt,label) rows."""
    X = []
    for l, w, h, wt, *_ in rows:
        vol = l * w * h
        X.append([
            l, w, h, wt,
            vol,
            vol / 5000.0,                    # dim weight
            l / max(h, 0.001),               # aspect ratio
            wt / max(vol, 0.001),            # density
            l + w + h,                       # perimeter sum
            l / max(w, 0.001),               # length-width ratio
        ])
    return np.array(X)


def _build_features_single(l: float, w: float, h: float, wt: float) -> np.ndarray:
    vol = l * w * h
    return np.array([[
        l, w, h, wt,
        vol,
        vol / 5000.0,
        l / max(h, 0.001),
        wt / max(vol, 0.001),
        l + w + h,
        l / max(w, 0.001),
    ]])


def _train():
    """
    Train all 5 models on 4800 synthetic samples.
    Each model is tuned independently for best accuracy.
    """
    logger.info("[ML] Training 5 models on 4800 synthetic samples...")
    rows   = _generate_data(4800)
    X_raw  = _build_features_batch(rows)
    labels = [r[4] for r in rows]

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    sc = StandardScaler()
    X  = sc.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models_to_train = [
        ("random_forest",
         RandomForestClassifier(n_estimators=150, max_depth=None,
                                min_samples_split=2, n_jobs=-1, random_state=42)),
        ("gradient_boost",
         GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                    max_depth=5, random_state=42)),
        ("extra_trees",
         ExtraTreesClassifier(n_estimators=150, n_jobs=-1, random_state=42)),
        ("svm",
         SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
        ("voting_ensemble",
         VotingClassifier(
             estimators=[
                 ("rf",  RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
                 ("gb",  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
                 ("et",  ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
             ],
             voting="soft", n_jobs=-1,
         )),
    ]

    results = {}
    for name, model in models_to_train:
        logger.info(f"[ML] Training {name}...")
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = acc
        _models[name] = model
        logger.info(f"[ML] {name}: accuracy={acc:.4f}")

    _models["label_encoder"] = le
    _models["scaler"]        = sc

    best = max(results, key=results.get)
    logger.info(
        f"[ML] All 5 models trained. "
        f"Best: {best} ({results[best]:.4f}). "
        f"Models: {list(results.keys())}"
    )


def load_models():
    """
    Load from disk if available, else auto-train in memory.
    Ensures all 5 models are always ready.
    """
    if not _sklearn_available:
        logger.info("[ML] sklearn not available — skipping")
        return

    import joblib
    model_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../../../ml_engine/models")
    )

    disk_files = {
        "random_forest":   os.path.join(model_dir, "rf_model.pkl"),
        "gradient_boost":  os.path.join(model_dir, "gb_model.pkl"),
        "extra_trees":     os.path.join(model_dir, "et_model.pkl"),
        "svm":             os.path.join(model_dir, "svm_model.pkl"),
        "voting_ensemble": os.path.join(model_dir, "ensemble_model.pkl"),
        "label_encoder":   os.path.join(model_dir, "label_encoder.pkl"),
        "scaler":          os.path.join(model_dir, "scaler.pkl"),
    }

    loaded = 0
    for name, path in disk_files.items():
        if os.path.exists(path):
            try:
                _models[name] = joblib.load(path)
                loaded += 1
                logger.info(f"[ML] Loaded from disk: {name}")
            except Exception as e:
                logger.warning(f"[ML] Could not load {name}: {e}")

    if loaded >= 5:
        logger.info(f"[ML] All {loaded} models loaded from disk")
        return

    logger.info(f"[ML] Only {loaded} disk models found — auto-training all 5 in memory")
    try:
        _train()
    except Exception as e:
        logger.error(f"[ML] Auto-train failed: {e}")
        import traceback
        traceback.print_exc()


def _validate_inputs(l: float, w: float, h: float, wt: float):
    errors = []
    if not (0.1 <= l  <= 200): errors.append(f"length {l} not in [0.1, 200]")
    if not (0.1 <= w  <= 200): errors.append(f"width {w} not in [0.1, 200]")
    if not (0.1 <= h  <= 200): errors.append(f"height {h} not in [0.1, 200]")
    if not (0.01 <= wt <= 100): errors.append(f"weight {wt} not in [0.01, 100]")
    if errors:
        raise ValueError(f"Invalid inputs: {'; '.join(errors)}")


def predict_all_models(l: float, w: float, h: float, wt: float) -> Dict[str, Any]:
    """
    Run ALL 5 models independently on this order.
    Returns individual predictions + confidence-weighted majority vote.

    Returns:
        {
            "voted_box":     str,           # final answer — majority vote
            "vote_confidence": float,       # avg confidence of winning models
            "model_votes": {               # what each model said
                "random_forest":   {"box": str, "confidence": float},
                "gradient_boost":  {"box": str, "confidence": float},
                "extra_trees":     {"box": str, "confidence": float},
                "svm":             {"box": str, "confidence": float},
                "voting_ensemble": {"box": str, "confidence": float},
            },
            "agreement": float,            # fraction of models that agreed
            "models_used": int,            # how many models ran
        }
    """
    if not _sklearn_available:
        raise RuntimeError("scikit-learn not installed")
    if len(_models) < 3:
        raise RuntimeError("Models not loaded yet")

    _validate_inputs(l, w, h, wt)

    feat_raw = _build_features_single(l, w, h, wt)
    feat     = _models["scaler"].transform(feat_raw) if "scaler" in _models else feat_raw
    le       = _models.get("label_encoder")

    model_names = ["random_forest", "gradient_boost", "extra_trees", "svm", "voting_ensemble"]
    votes        = {}   # box_type -> total confidence score
    model_votes  = {}
    models_run   = 0

    for name in model_names:
        if name not in _models:
            continue
        try:
            model  = _models[name]
            pred   = model.predict(feat)[0]
            proba  = model.predict_proba(feat)[0]
            conf   = float(np.max(proba))
            box    = le.inverse_transform([pred])[0] if le else str(pred)

            model_votes[name] = {"box": box, "confidence": round(conf, 4)}
            votes[box]        = votes.get(box, 0.0) + conf
            models_run       += 1

            logger.info(
                f"[ML] {name}: {box} (conf={conf:.1%}) "
                f"dims={l}x{w}x{h}cm wt={wt}kg"
            )
        except Exception as e:
            logger.warning(f"[ML] {name} failed: {e}")

    if not votes:
        raise RuntimeError("All models failed")

    # Confidence-weighted majority vote
    voted_box      = max(votes, key=votes.get)
    total_conf     = sum(votes.values())
    vote_confidence = round(votes[voted_box] / total_conf, 4) if total_conf > 0 else 0.0
    agreement      = round(
        sum(1 for v in model_votes.values() if v["box"] == voted_box) / models_run, 4
    )

    logger.info(
        f"[ML] VOTE RESULT: {voted_box} "
        f"(agreement={agreement:.0%}, conf={vote_confidence:.0%}, "
        f"{models_run} models)"
    )

    return {
        "voted_box":       voted_box,
        "vote_confidence": vote_confidence,
        "model_votes":     model_votes,
        "agreement":       agreement,
        "models_used":     models_run,
    }


def predict_packaging(l: float, w: float, h: float, wt: float) -> Dict[str, Any]:
    """
    Backwards-compatible single prediction — uses all 5 models internally.
    """
    result = predict_all_models(l, w, h, wt)
    return {
        "recommended_box":  result["voted_box"],
        "confidence_score": result["vote_confidence"],
        "model_used":       f"5-model-vote (agreement={result['agreement']:.0%})",
        "all_votes":        result,
    }


def get_loaded_models() -> List[str]:
    return [k for k in _models if k not in ("label_encoder", "scaler")]


def is_ml_available() -> bool:
    predictors = [k for k in _models if k not in ("label_encoder", "scaler")]
    return _sklearn_available and len(predictors) >= 3
