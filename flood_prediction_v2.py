"""
flood_prediction_v2.py
========================
A genuine BEFORE-THE-EVENT flood forecast, fixing the limitation in
flood_risk_real.py (which could only compare "flood-prone" locations
against "right now" readings, not a true precursor signal).

REAL DATA THROUGHOUT:
  - EVENTS: real historical flood events with real dates and real
    locations, from GLOBAL_FLOOD_DB/MODIS_EVENTS/V1 -- Dartmouth Flood
    Observatory data, hosted directly in Earth Engine (913 real global
    events, 2000-2018; built by Cloud to Street / Google).
  - POSITIVE FEATURE: real NDWI trend for the 30 days BEFORE each
    event's actual start date, at that event's actual location --
    a genuine precursor signal, not a same-day/present-day reading.
  - NEGATIVE FEATURE: the SAME real location, but the 30 days before a
    DIFFERENT date (180 days earlier) with no recorded flood event --
    a real non-flood control period at the same place, not synthetic.

CONSTRAINT: restricted to events after Sentinel-2's 2015 launch, since
NDWI here is computed from Sentinel-2 -- this is an honest limitation
(older events can't be paired with Sentinel-2 data), not an omission.

Usage:
    python flood_prediction_v2.py --n_events 20 --days_back 30
"""

import argparse
import json
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def get_real_flood_events(n_events, min_date="2016-06-23"):
    """
    Pull real flood events from the Global Flood Database, each with a
    real (lat, lon, date). min_date defaults to one full year after
    Sentinel-2's 2015-06-23 launch, not the launch date itself -- events
    right at the boundary have 30-day "before" windows that reach back
    into the pre-launch gap with no usable imagery at all.
    """
    from gee_pipeline import _require_ee
    ee = _require_ee()

    collection = (
        ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")
        .filterDate(min_date, "2018-12-10")
        .limit(n_events)
    )
    info = collection.getInfo()

    events = []
    for feature in info["features"]:
        props = feature["properties"]
        # system:time_start is milliseconds since epoch -- standard GEE
        # temporal property, present on every image in the collection.
        event_date = datetime.fromtimestamp(props["system:time_start"] / 1000, tz=timezone.utc)
        events.append({
            "id": props.get("id"),
            "lat": props["dfo_centroid_y"],
            "lon": props["dfo_centroid_x"],
            "date": event_date.strftime("%Y-%m-%d"),
            "country": props.get("dfo_country", "unknown"),
        })
    return events


def build_dataset(events, days_back):
    """
    For each real event: pull real pre-event NDWI trend (positive
    sample) and real pre-control-date NDWI trend at the SAME real
    location, 180 days earlier (negative sample, assumed non-flood).
    """
    from gee_pipeline import get_ndwi_before_date

    rows = []
    for ev in events:
        event_date = datetime.strptime(ev["date"], "%Y-%m-%d")
        control_date = event_date - timedelta(days=180)

        mean_pos, slope_pos = get_ndwi_before_date(ev["lat"], ev["lon"], ev["date"], days_back)
        if mean_pos is not None:
            print(f"  [FLOOD]  {ev['country']:<15} {ev['date']}  mean_ndwi={mean_pos:+.4f} slope={slope_pos:+.5f}")
            rows.append((mean_pos, slope_pos, 1))
        else:
            print(f"  Skipping positive sample for event {ev['id']} (no imagery in window)")

        mean_neg, slope_neg = get_ndwi_before_date(
            ev["lat"], ev["lon"], control_date.strftime("%Y-%m-%d"), days_back
        )
        if mean_neg is not None:
            print(f"  [NORMAL] {ev['country']:<15} {control_date.strftime('%Y-%m-%d')}  mean_ndwi={mean_neg:+.4f} slope={slope_neg:+.5f}")
            rows.append((mean_neg, slope_neg, 0))
        else:
            print(f"  Skipping negative sample for event {ev['id']} (no imagery in window)")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_events", type=int, default=20)
    parser.add_argument("--days_back", type=int, default=30,
                         help="How many days before each date to compute the NDWI trend over")
    args = parser.parse_args()

    print(f"Pulling {args.n_events} real flood events from GLOBAL_FLOOD_DB...")
    events = get_real_flood_events(args.n_events)
    print(f"Got {len(events)} real events\n")

    print("Building real precursor-NDWI dataset (this makes real GEE calls, may take a few minutes)...")
    rows = build_dataset(events, args.days_back)

    if len(rows) < 10:
        print(f"\nOnly {len(rows)} usable samples -- try more --n_events.")
        return

    X = np.array([[r[0], r[1]] for r in rows])  # mean_ndwi, trend_slope
    y = np.array([r[2] for r in rows])

    print(f"\nTotal samples: {len(y)} (flood={int(y.sum())}, normal={int((1-y).sum())})")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\nFlood prediction -- REAL pre-event precursor signal, real historical events")
    print(f"  Test set size: {len(y_test)}")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 score:  {f1:.3f}")
    print(f"  Coefficients: mean_ndwi={model.coef_[0][0]:+.3f}, trend_slope={model.coef_[0][1]:+.3f}")

    tn, fp, fn, tp = cm.ravel()
    with open("results_flood_prediction_v2.json", "w") as f:
        json.dump({
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "feature_names": ["mean_ndwi", "trend_slope"],
            "feature_weights": [float(w) for w in model.coef_[0]],
            "n_events_pulled": len(events), "n_samples": len(y),
        }, f, indent=2)
    print("\nSaved: results_flood_prediction_v2.json")


if __name__ == "__main__":
    main()