"""
ML Training Pipeline — AI Packaging Automation Platform
Trains 4 individual models + 1 voting ensemble.

Models trained:
  1. Random Forest Classifier
  2. Gradient Boosting Classifier
  3. Extra Trees Classifier
  4. Support Vector Machine (SVM)
  5. Voting Ensemble (RF + GB + ET)

Run: python train_models.py
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

os.makedirs("models", exist_ok=True)


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve model accuracy."""
    df = df.copy()
    df["volume"]       = df["length"] * df["width"] * df["height"]
    df["dim_weight"]   = df["volume"] / 5000.0
    df["aspect_ratio"] = df["length"] / df["height"].replace(0, 0.001)
    df["density"]      = df["weight"] / df["volume"].replace(0, 0.001)
    df["lwh_sum"]      = df["length"] + df["width"] + df["height"]
    df["lw_ratio"]     = df["length"] / df["width"].replace(0, 0.001)
    return df


FEATURE_COLS = ["length", "width", "height", "weight", "volume",
                "dim_weight", "aspect_ratio", "density", "lwh_sum", "lw_ratio"]


# ── Dataset Generation (if no CSV provided) ───────────────────────────────────

def generate_synthetic_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate a realistic synthetic packaging dataset.
    Box assignment rules mirror real Indian ecommerce logistics.

    Box sizes (cm):
      Box_XS  — 15×10×10 — small accessories, SIM cards
      Box_S   — 25×20×15 — books, small electronics
      Box_M   — 35×25×20 — clothing, small appliances
      Box_L   — 50×40×30 — shoes, medium gadgets
      Box_XL  — 70×50×40 — large appliances, bulky items
      Box_XXL — 90×70×60 — very large or heavy items
    """
    np.random.seed(42)
    records = []

    box_configs = [
        ("Box_XS",  (5,  15),  (5,  10),  (5,  10),  (0.05, 0.5)),
        ("Box_S",   (15, 25),  (10, 20),  (5,  15),  (0.3,  3.0)),
        ("Box_M",   (25, 40),  (15, 30),  (10, 25),  (1.0,  8.0)),
        ("Box_L",   (35, 55),  (25, 45),  (20, 35),  (3.0,  15.0)),
        ("Box_XL",  (50, 75),  (35, 55),  (30, 50),  (8.0,  25.0)),
        ("Box_XXL", (70, 100), (50, 80),  (50, 70),  (15.0, 40.0)),
    ]

    samples_per_box = n_samples // len(box_configs)

    for box_type, l_range, w_range, h_range, wt_range in box_configs:
        for _ in range(samples_per_box):
            length = np.random.uniform(*l_range)
            width  = np.random.uniform(*w_range)
            height = np.random.uniform(*h_range)
            weight = np.random.uniform(*wt_range)

            # Add noise to make the dataset realistic
            length += np.random.normal(0, 1.5)
            width  += np.random.normal(0, 1.2)
            height += np.random.normal(0, 1.0)
            weight += np.random.normal(0, 0.3)

            # Clip to positive values
            length = max(1.0, length)
            width  = max(1.0, width)
            height = max(1.0, height)
            weight = max(0.05, weight)

            records.append({
                "length":   round(length, 2),
                "width":    round(width, 2),
                "height":   round(height, 2),
                "weight":   round(weight, 3),
                "box_type": box_type,
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("dataset/packaging_dataset.csv", index=False)
    print(f"[data] Generated {len(df)} samples → dataset/packaging_dataset.csv")
    return df


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_dataset(path: str = "dataset/packaging_dataset.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[data] No dataset found at {path}, generating synthetic data...")
        os.makedirs("dataset", exist_ok=True)
        return generate_synthetic_dataset()

    df = pd.read_csv(path)
    print(f"[data] Loaded {len(df)} rows from {path}")

    # Clean invalid rows
    required = ["length", "width", "height", "weight", "box_type"]
    df = df.dropna(subset=required)
    df = df[(df["length"] > 0) & (df["width"] > 0) &
            (df["height"] > 0) & (df["weight"] > 0)]
    print(f"[data] After cleaning: {len(df)} valid rows")
    return df


# ── Training ──────────────────────────────────────────────────────────────────

def train_all_models():
    print("\n" + "="*60)
    print("  PackAI ML Training Pipeline")
    print("="*60)

    # Load and prepare data
    df = load_dataset()
    df = engineer_features(df)

    X = df[FEATURE_COLS].values
    y = df["box_type"].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"\n[data] Classes: {list(le.classes_)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"[data] Train: {len(X_train)}, Test: {len(X_test)}")

    results = {}

    # ── Model 1: Random Forest ────────────────────────────────────────────────
    print("\n[train] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators = 200,
        max_depth    = None,
        min_samples_split = 2,
        n_jobs       = -1,
        random_state = 42,
    )
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    results["Random Forest"] = rf_acc
    print(f"[train] Random Forest accuracy: {rf_acc:.4f}")
    joblib.dump(rf, "models/rf_model.pkl")

    # ── Model 2: Gradient Boosting ────────────────────────────────────────────
    print("\n[train] Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators = 200,
        learning_rate = 0.1,
        max_depth    = 5,
        random_state = 42,
    )
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    results["Gradient Boosting"] = gb_acc
    print(f"[train] Gradient Boosting accuracy: {gb_acc:.4f}")
    joblib.dump(gb, "models/gb_model.pkl")

    # ── Model 3: Extra Trees ──────────────────────────────────────────────────
    print("\n[train] Training Extra Trees...")
    et = ExtraTreesClassifier(
        n_estimators = 200,
        n_jobs       = -1,
        random_state = 42,
    )
    et.fit(X_train, y_train)
    et_acc = accuracy_score(y_test, et.predict(X_test))
    results["Extra Trees"] = et_acc
    print(f"[train] Extra Trees accuracy: {et_acc:.4f}")
    joblib.dump(et, "models/et_model.pkl")

    # ── Model 4: SVM ──────────────────────────────────────────────────────────
    print("\n[train] Training SVM...")
    svm = SVC(kernel="rbf", probability=True, C=10, gamma="scale", random_state=42)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    results["SVM"] = svm_acc
    print(f"[train] SVM accuracy: {svm_acc:.4f}")
    joblib.dump(svm, "models/svm_model.pkl")

    # ── Model 5: Voting Ensemble ──────────────────────────────────────────────
    print("\n[train] Training Voting Ensemble (RF + GB + ET)...")
    ensemble = VotingClassifier(
        estimators = [
            ("rf",  RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)),
            ("gb",  GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)),
            ("et",  ExtraTreesClassifier(n_estimators=150, n_jobs=-1, random_state=42)),
        ],
        voting = "soft",   # use predicted probabilities
        n_jobs = -1,
    )
    ensemble.fit(X_train, y_train)
    ens_acc = accuracy_score(y_test, ensemble.predict(X_test))
    results["Ensemble"] = ens_acc
    print(f"[train] Ensemble accuracy: {ens_acc:.4f}")
    joblib.dump(ensemble, "models/ensemble_model.pkl")

    # Save shared artifacts
    joblib.dump(le,     "models/label_encoder.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    # ── Results summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Training Complete — Accuracy Summary")
    print("="*60)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 40)
        print(f"  {name:<22} {acc:.4f}  {bar}")

    best_model = max(results, key=results.get)
    print(f"\n  Best model: {best_model} ({results[best_model]:.4f})")
    print(f"  All models saved to: models/")
    print("="*60 + "\n")

    # Detailed report for best individual model
    best_map = {
        "Random Forest":     rf,
        "Gradient Boosting": gb,
        "Extra Trees":       et,
        "SVM":               svm,
        "Ensemble":          ensemble,
    }
    best = best_map.get(best_model, ensemble)
    y_pred = best.predict(X_test)
    print("Classification Report (best model):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))


if __name__ == "__main__":
    train_all_models()
