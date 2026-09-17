"""
wildfire_risk_prediction.py
=============================
DISASTER PREDICTION for wildfire (Technical Task 4) -- estimate wildfire
risk from precursor weather conditions, trained on REAL historical fire
records (not synthetic data).

DATA SOURCE: the Algerian Forest Fires dataset (UCI Machine Learning
Repository, id=547). 244 real daily records from two regions of Algeria
(Bejaia, Sidi Bel-Abbes), June-September 2012, each with real weather
observations and a real ground-truth label of whether a fire occurred.
Citation: Abid, F. (2019). Algerian Forest Fires [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C5KW4N

HONEST SCOPE NOTE: 244 records from ONE season in ONE country is a
small, narrow sample -- real and trained/evaluated, but generalizing
beyond Algeria/summer conditions should be done cautiously.

Usage:
    pip install ucimlrepo scikit-learn
    python wildfire_risk_prediction.py
"""

import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)


def _to_float(cell):
    """Robustly parse one cell to a float (handles a few malformed raw entries)."""
    match = re.match(r"^\s*(-?\d+\.?\d*)", str(cell))
    return float(match.group(1)) if match else np.nan


def load_real_data():
    """Pulls the real Algerian Forest Fires dataset directly from the UCI ML Repository."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=547)
    X_raw = dataset.data.features.copy()
    y_raw = dataset.data.targets.copy()

    feature_cols = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
    feature_cols = [c for c in feature_cols if c in X_raw.columns]
    X = X_raw[feature_cols].apply(lambda col: col.map(_to_float))

    label_col = y_raw.columns[0]
    y = y_raw[label_col].astype(str).str.strip().str.lower().map(
        lambda v: 1 if "not" not in v else 0
    )

    valid = X.notna().all(axis=1)
    n_dropped = (~valid).sum()
    if n_dropped:
        print(f"Dropped {n_dropped} row(s) with malformed values in the raw dataset")
    X, y = X[valid], y[valid]

    return X.values, y.values, feature_cols


def live_fuel_dryness_check(lat, lon):
    """
    Live, present-day vegetation dryness (NDMI) for a real location right
    now, pulled from Sentinel-2 via Earth Engine. Deliberately kept
    separate from the trained model's features (see module docstring
    discussion: training data is from 2012, Sentinel-2 launched 2015).
    """
    from gee_pipeline import get_current_ndmi

    ndmi = get_current_ndmi(lat, lon)
    if ndmi is None:
        print(f"No cloud-free Sentinel-2 image found near ({lat}, {lon}) in the last 15 days.")
        return None
    print(f"Live NDMI (fuel moisture) at ({lat}, {lon}): {ndmi:.4f}")
    if ndmi < 0:
        print("  -> vegetation reads DRY (negative NDMI) -- elevated fuel availability")
    elif ndmi < 0.2:
        print("  -> vegetation moisture is low-moderate")
    else:
        print("  -> vegetation reads moist -- lower fuel availability")
    return ndmi


def _save_results(accuracy, precision, recall, f1, cm, feature_names, weights):
    """Save real run results to a file, so report charts stay reproducible."""
    import json
    tn, fp, fn, tp = cm.ravel().tolist()
    data = {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "feature_names": list(feature_names),
        "feature_weights": [float(w) for w in weights],
    }
    with open("results_wildfire_prediction.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Saved: results_wildfire_prediction.json")


def main():
    X, y, feature_names = load_real_data()
    print(f"Loaded {len(y)} REAL records from the Algerian Forest Fires dataset")
    print(f"  Fire days:     {int(y.sum())}")
    print(f"  Non-fire days: {int((1 - y).sum())}")
    print(f"  Features used: {feature_names}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Wildfire risk prediction -- trained on REAL historical data")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 score:  {f1:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion matrix (test set, n={len(y_test)}):")
    print(f"  True negatives:  {tn}   False positives: {fp}")
    print(f"  False negatives: {fn}   True positives:  {tp}")

    print("\nFeature weights (standardized -- higher magnitude = more influence):")
    sorted_pairs = sorted(zip(feature_names, model.coef_[0]), key=lambda x: -abs(x[1]))
    for name, weight in sorted_pairs:
        print(f"  {name:<15}{weight:+.3f}")

    print(
        "\nNote: small sample (244 records, one season, one country) -- "
        "treat these metrics as evidence the pipeline works on real data, "
        "not as a validated production-ready predictor."
    )

    _save_results(accuracy, precision, recall, f1, cm, feature_names, model.coef_[0])

    # Uncomment for live, real-time vegetation dryness at an actual location
    # (requires ee.Authenticate()/ee.Initialize() to have run):
    # live_fuel_dryness_check(lat=34.05, lon=-118.5)  # LA area


if __name__ == "__main__":
    main()
