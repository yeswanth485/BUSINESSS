"""
ML Service — FALLBACK ONLY.

Rules (as required):
  1. ML is fallback — only called when FFD rule engine returns no valid box
  2. ML does NOT override rule-based decisions
  3. All inputs validated before prediction (ranges, types, sanity checks)
  4. Every prediction logged: model name, confidence, inputs, result, latency

Models (5, all trained and ready):
  1. Random Forest       — fast, handles noise
  2. Gradient Boosting   — high accuracy on structured data
  3. Extra Trees         — fast, diverse from RF
  4. SVM (RBF kernel)    — strong on small dimensions
  5. Voting Ensemble     — RF+GB+ET soft voting

Flow:
  FFD runs first → valid box found → return (ML never called)
  FFD fails      → ML predicts     → result logged and returned
"""
import os
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

_models: Dict[str, Any] = {}
_sklearn_available = False
_prediction_count  = 0   # total predictions made this session

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
    from sklearn.metrics import accuracy_score, classification_report
    _sklearn_available = True
    logger.info("[ML] scikit-learn available")
except ImportError:
    logger.warning("[ML] scikit-learn not installed — FFD handles all orders")


# ── Input validation constants ────────────────────────────────────────────────
VALID_RANGES = {
    "length": (0.1,  200.0),
    "width":  (0.1,  200.0),
    "height": (0.1,  200.0),
    "weight": (0.01, 100.0),
}

# Valid box types this service can predict
VALID_BOXES = ["Box_XS", "Box_S", "Box_M", "Box_L", "Box_XL", "Box_XXL"]

# Box catalogue for synthetic training data
BOX_CONFIGS = [
    ("Box_XS",  (5, 15),   (5, 10),   (5, 10),   (0.05, 0.5)),
    ("Box_S",   (15, 25),  (10, 20),  (5, 15),   (0.3,  3.0)),
    ("Box_M",   (25, 40),  (15, 30),  (10, 25),  (1.0,  8.0)),
    ("Box_L",   (35, 55),  (25, 45),  (20, 35),  (3.0,  15.0)),
    ("Box_XL",  (50, 75),  (35, 55),  (30, 50),  (8.0,  25.0)),
    ("Box_XXL", (70, 100), (50, 80),  (50, 70),  (15.0, 40.0)),
]


# ── Input validation ──────────────────────────────────────────────────────────
def validate_inputs(
    l: float, w: float, h: float, wt: float
) -> Tuple[bool, List[str]]:
    """
    Validate dimensions and weight before prediction.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    inputs = {"length": l, "width": w, "height": h, "weight": wt}
    for name, value in inputs.items():
        # Type check
        try:
            v = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name}={value!r} is not a number")
            continue
        # NaN / Inf check
        if np.isnan(v) or np.isinf(v):
            errors.append(f"{name}={value} is NaN or Inf")
            continue
        # Range check
        lo, hi = VALID_RANGES[name]
        if not (lo <= v <= hi):
            errors.append(f"{name}={v} out of range [{lo}, {hi}]")
    return (len(errors) == 0, errors)


# ── Feature engineering ───────────────────────────────────────────────────────
def _build_features(l: float, w: float, h: float, wt: float) -> np.ndarray:
    vol = l * w * h
    return np.array([[
        l, w, h, wt,
        vol,
        vol / 5000.0,                # dim weight
        l / max(h, 0.001),           # aspect ratio (length/height)
        wt / max(vol, 0.001),        # density kg/cm³
        l + w + h,                   # perimeter sum
        l / max(w, 0.001),           # length-width ratio
    ]])


def _build_features_batch(rows) -> np.ndarray:
    X = []
    for l, w, h, wt, *_ in rows:
        vol = l * w * h
        X.append([l, w, h, wt, vol, vol/5000.0,
                  l/max(h,0.001), wt/max(vol,0.001), l+w+h, l/max(w,0.001)])
    return np.array(X)


# ── Training data generation ──────────────────────────────────────────────────
def _generate_data(n: int = 4800):
    """Synthetic Indian ecommerce samples — 800 per box type."""
    rng  = np.random.RandomState(42)
    rows = []
    per  = n // len(BOX_CONFIGS)
    for box_type, lr, wr, hr, wtr in BOX_CONFIGS:
        for _ in range(per):
            l  = float(np.clip(rng.uniform(*lr)  + rng.normal(0, 1.5), 0.1, 200))
            w  = float(np.clip(rng.uniform(*wr)  + rng.normal(0, 1.2), 0.1, 200))
            h  = float(np.clip(rng.uniform(*hr)  + rng.normal(0, 1.0), 0.1, 200))
            wt = float(np.clip(rng.uniform(*wtr) + rng.normal(0, 0.3), 0.01, 100))
            rows.append((round(l,2), round(w,2), round(h,2), round(wt,3), box_type))
    rng.shuffle(rows)
    return rows


# ── Model training ────────────────────────────────────────────────────────────
def _train():
    logger.info("[ML] Auto-training 5 models on 4800 synthetic samples...")
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

    models_def = [
        ("random_forest",
         RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)),
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
                 ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
                 ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
                 ("et", ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
             ],
             voting="soft", n_jobs=-1,
         )),
    ]

    accuracies = {}
    for name, model in models_def:
        t0 = time.time()
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        elapsed = round(time.time() - t0, 2)
        accuracies[name] = acc
        _models[name] = model
        logger.info(f"[ML] {name}: accuracy={acc:.4f} trained_in={elapsed}s")

    _models["label_encoder"] = le
    _models["scaler"]        = sc

    best = max(accuracies, key=accuracies.get)
    logger.info(
        f"[ML] Training complete. Best={best} ({accuracies[best]:.4f}). "
        f"All: {', '.join(f'{k}={v:.4f}' for k,v in accuracies.items())}"
    )


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models():
    """Load from disk if available, else auto-train in memory."""
    if not _sklearn_available:
        logger.info("[ML] sklearn not available — FFD handles all orders")
        return

    import joblib
    model_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../../../ml_engine/models")
    )
    disk_map = {
        "random_forest":   "rf_model.pkl",
        "gradient_boost":  "gb_model.pkl",
        "extra_trees":     "et_model.pkl",
        "svm":             "svm_model.pkl",
        "voting_ensemble": "ensemble_model.pkl",
        "label_encoder":   "label_encoder.pkl",
        "scaler":          "scaler.pkl",
    }
    loaded = 0
    for name, fname in disk_map.items():
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            try:
                _models[name] = joblib.load(path)
                loaded += 1
                logger.info(f"[ML] Loaded from disk: {name}")
            except Exception as e:
                logger.warning(f"[ML] Disk load failed for {name}: {e}")

    if loaded >= 5:
        logger.info(f"[ML] {loaded} models loaded from disk — ready")
        return

    logger.info(f"[ML] {loaded} disk models — auto-training all 5...")
    try:
        _train()
    except Exception as e:
        logger.error(f"[ML] Auto-train failed: {e}")
        import traceback; traceback.print_exc()


# ── FALLBACK PREDICTION (called only when FFD fails) ─────────────────────────
def predict_fallback(
    l: float, w: float, h: float, wt: float,
    order_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    FALLBACK ONLY — called by decision_service when FFD finds no valid box.

    Steps:
      1. Validate inputs — reject invalid dimensions immediately
      2. Run all available models
      3. Confidence-weighted majority vote
      4. Log full prediction details
      5. Return result with confidence and model breakdown

    Raises:
      ValueError — invalid inputs (caller should use rule-based fallback)
      RuntimeError — models not loaded
    """
    global _prediction_count

    # Step 1: Validate
    is_valid, errors = validate_inputs(l, w, h, wt)
    if not is_valid:
        logger.error(
            f"[ML-FALLBACK] INVALID INPUT order={order_id} "
            f"dims={l}x{w}x{h}cm wt={wt}kg errors={errors}"
        )
        raise ValueError(f"ML input validation failed: {'; '.join(errors)}")

    if not _sklearn_available:
        raise RuntimeError("scikit-learn not installed")
    if len(_models) < 3:
        raise RuntimeError("ML models not loaded")

    t_start = time.time()
    _prediction_count += 1
    pred_id = _prediction_count

    # Step 2: Build features
    feat_raw = _build_features(l, w, h, wt)
    sc = _models.get("scaler")
    le = _models.get("label_encoder")
    feat = sc.transform(feat_raw) if sc is not None else feat_raw

    # Step 3: Run all models
    model_names = ["random_forest", "gradient_boost", "extra_trees", "svm", "voting_ensemble"]
    votes:       Dict[str, float] = {}
    model_votes: Dict[str, Dict]  = {}
    models_run   = 0

    for name in model_names:
        if name not in _models:
            continue
        try:
            model = _models[name]
            pred  = model.predict(feat)[0]
            proba = model.predict_proba(feat)[0]
            conf  = float(np.max(proba))
            box   = le.inverse_transform([pred])[0] if le else str(pred)

            # Validate the box name returned
            if box not in VALID_BOXES:
                logger.warning(f"[ML-FALLBACK] {name} returned unknown box: {box}")
                continue

            model_votes[name] = {"box": box, "confidence": round(conf, 4)}
            votes[box]        = votes.get(box, 0.0) + conf
            models_run       += 1

        except Exception as e:
            logger.warning(f"[ML-FALLBACK] {name} predict failed: {e}")

    if not votes:
        raise RuntimeError("All ML models failed during fallback prediction")

    # Step 4: Confidence-weighted majority vote
    voted_box      = max(votes, key=votes.get)
    total_conf     = sum(votes.values())
    vote_confidence = round(votes[voted_box] / total_conf, 4) if total_conf > 0 else 0.0
    agreement      = round(
        sum(1 for v in model_votes.values() if v["box"] == voted_box) / models_run, 4
    ) if models_run > 0 else 0.0
    latency_ms     = round((time.time() - t_start) * 1000, 2)

    # Step 5: Log full prediction
    logger.info(
        f"[ML-FALLBACK] pred_id={pred_id} order={order_id} "
        f"result={voted_box} conf={vote_confidence:.1%} "
        f"agreement={agreement:.0%} models={models_run} "
        f"dims={l}x{w}x{h}cm wt={wt}kg latency={latency_ms}ms"
    )
    for name, v in model_votes.items():
        logger.debug(
            f"[ML-FALLBACK]   {name}: {v['box']} ({v['confidence']:.1%})"
        )

    return {
        "recommended_box":  voted_box,
        "confidence_score": vote_confidence,
        "agreement":        agreement,
        "models_used":      models_run,
        "model_votes":      model_votes,
        "latency_ms":       latency_ms,
        "pred_id":          pred_id,
        "role":             "fallback",   # never "primary" — FFD is primary
    }


# ── Batch fallback prediction ─────────────────────────────────────────────────
def predict_batch(orders: List[tuple]) -> List[Dict[str, Any]]:
    """
    Batch fallback prediction — only called for orders where FFD failed.
    Each tuple: (length, width, height, weight).
    Validates all inputs first. Runs all 5 models in one matrix pass.
    """
    if not _sklearn_available or len(_models) < 3:
        raise RuntimeError("ML models not available")
    if not orders:
        return []

    # Validate all inputs before any computation
    validation_errors = []
    for i, (l, w, h, wt) in enumerate(orders):
        ok, errs = validate_inputs(float(l), float(w), float(h), float(wt))
        if not ok:
            validation_errors.append(f"order[{i}]: {'; '.join(errs)}")

    if validation_errors:
        logger.error(f"[ML-Batch] Validation failed: {validation_errors}")
        raise ValueError(f"Batch validation errors: {validation_errors}")

    sc = _models.get("scaler")
    le = _models.get("label_encoder")

    X_raw = []
    for l, w, h, wt in orders:
        l, w, h, wt = float(l), float(w), float(h), float(wt)
        vol = l * w * h
        X_raw.append([l, w, h, wt, vol, vol/5000.0,
                       l/max(h,0.001), wt/max(vol,0.001), l+w+h, l/max(w,0.001)])
    X = np.array(X_raw)
    if sc is not None:
        X = sc.transform(X)

    model_names = ["random_forest", "gradient_boost", "extra_trees", "svm", "voting_ensemble"]
    all_preds  = {}
    all_probas = {}
    for name in model_names:
        if name not in _models:
            continue
        try:
            all_preds[name]  = _models[name].predict(X)
            all_probas[name] = _models[name].predict_proba(X)
        except Exception as e:
            logger.warning(f"[ML-Batch] {name} failed: {e}")

    if not all_preds:
        raise RuntimeError("All models failed in batch prediction")

    run_models = list(all_preds.keys())
    results = []
    for i in range(len(orders)):
        votes = {}
        for name in run_models:
            pred = all_preds[name][i]
            conf = float(np.max(all_probas[name][i]))
            box  = le.inverse_transform([pred])[0] if le else str(pred)
            if box not in VALID_BOXES:
                continue
            votes[box] = votes.get(box, 0.0) + conf

        if not votes:
            # Fallback to largest box if all predictions invalid
            votes = {"Box_XXL": 1.0}

        voted_box  = max(votes, key=votes.get)
        total_conf = sum(votes.values())
        agreement  = sum(
            1 for name in run_models
            if (le.inverse_transform([all_preds[name][i]])[0] if le else str(all_preds[name][i])) == voted_box
        ) / len(run_models)

        results.append({
            "voted_box":       voted_box,
            "vote_confidence": round(votes[voted_box] / total_conf, 4) if total_conf else 0.0,
            "agreement":       round(agreement, 4),
            "models_used":     len(run_models),
            "role":            "fallback",
        })

    logger.info(
        f"[ML-Batch] {len(orders)} fallback predictions, "
        f"{len(run_models)} models, role=fallback"
    )
    return results


# ── Consistency test ──────────────────────────────────────────────────────────
def run_consistency_test() -> Dict[str, Any]:
    """
    Test that models give consistent results across repeated calls.
    Runs the same inputs 3 times and checks outputs are identical.
    Returns test report.
    """
    if not is_ml_available():
        return {"passed": False, "reason": "ML not available"}

    test_cases = [
        (12.0, 8.0,  5.0,  0.15),   # earbuds → Box_XS
        (32.0, 22.0, 14.0, 0.85),   # shoes   → Box_M
        (28.0, 22.0, 35.0, 3.5),    # appliance → Box_L
    ]
    results = {}
    for dims in test_cases:
        key = f"{dims[0]}x{dims[1]}x{dims[2]}cm_{dims[3]}kg"
        runs = []
        for _ in range(3):
            r = predict_fallback(*dims)
            runs.append(r["recommended_box"])
        consistent = len(set(runs)) == 1
        results[key] = {"boxes": runs, "consistent": consistent}

    all_consistent = all(v["consistent"] for v in results.values())
    logger.info(f"[ML-Test] Consistency test: {'PASS' if all_consistent else 'FAIL'} — {results}")
    return {"passed": all_consistent, "details": results}


# ── Public helpers ────────────────────────────────────────────────────────────
def get_loaded_models() -> List[str]:
    return [k for k in _models if k not in ("label_encoder", "scaler")]


def is_ml_available() -> bool:
    predictors = [k for k in _models if k not in ("label_encoder", "scaler")]
    return _sklearn_available and len(predictors) >= 3


def get_prediction_count() -> int:
    """Total ML fallback predictions made this session."""
    return _prediction_count


# ── Backward compatibility ─────────────────────────────────────────────────────
def predict_all_models(l, w, h, wt):
    """Legacy alias — maps to predict_fallback."""
    r = predict_fallback(l, w, h, wt)
    return {
        "voted_box":       r["recommended_box"],
        "vote_confidence": r["confidence_score"],
        "agreement":       r["agreement"],
        "models_used":     r["models_used"],
        "model_votes":     r["model_votes"],
    }


def predict_packaging(l, w, h, wt):
    """Legacy alias — maps to predict_fallback."""
    r = predict_fallback(l, w, h, wt)
    return {
        "recommended_box":  r["recommended_box"],
        "confidence_score": r["confidence_score"],
        "model_used":       f"fallback-5-model-vote",
        "all_votes":        r,
    }
