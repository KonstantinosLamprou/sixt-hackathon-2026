"""Erstes trainiertes Modell: Gradient Boosting auf dem Kilometer-ZUWACHS seit dem letzten bekannten Stand.

Ablauf (pro Evaluations-Stichtag E):
  1. Trainingsbeispiele: viele frueher liegende Stichtage C (alle <= E - 133 Tage, damit das Ziel nie in die
     Evaluationszeit hineinragt). Fuer jeden C: Historie bis C -> Features, Ziel = tatsaechlicher Zuwachs.
  2. Nur Trainingsfahrzeuge (Holdout-Fahrzeuge werden NIE zum Training benutzt).
  3. Bewertung auf Holdout-Fahrzeugen am Stichtag E, zusammen mit den Baselines (gleiche Zeilen!).

Usage (aus Project_2/): python -m scripts.run_forecast
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import RESULTS
from src.data import SAMPLE, daily_series, load_raw, split_vehicles, vehicle_static
from src.baselines import CUTOFFS, HORIZON_DAYS, add_baselines, make_task
from src.metrics import by_horizon

TRAIN_START = "2022-07-01"
TRAIN_STEP_DAYS = 45
ROWS_PER_VEHICLE = 8          # Zielzeilen pro Fahrzeug und Trainings-Stichtag (Subsampling)
CAT_COLS = ["vehicle_level", "vehicle_type", "driven_wheels"]
NUM_STATIC = ["model_year", "power_ps", "cubic_capacity_liter", "vhcl_max_mileage", "acriss_code_anonymized"]
RATE_COLS = ["rate_all", "rate_30", "rate_90", "rate_180", "rate_fb"]


def build_features(t: pd.DataFrame, static: pd.DataFrame, cutoff: pd.Timestamp, cat_maps: dict) -> pd.DataFrame:
    """t = Ausgabe von add_baselines (enthaelt Historie-Infos + Ziel-Zeilen)."""
    t = t.join(static.drop(columns=["vehicle_level"], errors="ignore"), on="vhcl_anon_id")
    X = pd.DataFrame(index=t.index)
    X["gap"] = t.gap
    X["stale_days"] = (cutoff - t.last_date).dt.days             # wie alt ist die letzte Messung zum Stichtag
    X["age_days"] = (cutoff - t.first_date).dt.days              # wie lange kennen wir das Fahrzeug
    X["n_hist"] = t.n_hist
    X["last_km"] = t.last_km
    X["km_share_of_max"] = t.last_km / t.vhcl_max_mileage
    X["target_month"] = t.date.dt.month
    X["target_dow"] = t.date.dt.day_of_week
    for c in RATE_COLS:
        X[c] = t[c]
    for c in ["b_rate_all", "b_rate_30", "b_rate_90", "b_rate_180", "b_segment"]:
        X["exp_" + c[2:]] = t[c] - t.last_km                    # erwarteter Zuwachs laut Baseline
    for c in NUM_STATIC:
        X[c] = t[c]
    for c in CAT_COLS:
        X[c] = t[c].map(cat_maps[c]).fillna(-1)
    return X


def make_dataset(series, static, cutoff, vehicles, cat_maps, sample=False, seed=0):
    last, t = make_task(series, cutoff)
    t = t[t.vhcl_anon_id.isin(vehicles) & (t.gap > 0)]
    t = add_baselines(t, static)
    if sample:
        t = t.sample(frac=1, random_state=seed).groupby("vhcl_anon_id").head(ROWS_PER_VEHICLE)
    cutoff = pd.Timestamp(cutoff)
    return t, build_features(t, static, cutoff, cat_maps)


def main():
    raw = load_raw(SAMPLE)
    series = daily_series(raw)
    static = vehicle_static(raw)
    cat_maps = {c: {v: i for i, v in enumerate(sorted(static[c].dropna().unique()))} for c in CAT_COLS}
    train_ids, hold_ids = split_vehicles(series.vhcl_anon_id.unique())
    print(f"{len(train_ids)} Trainings-, {len(hold_ids)} Holdout-Fahrzeuge")

    out = []
    for E in CUTOFFS:
        last_train_cutoff = pd.Timestamp(E) - pd.Timedelta(days=HORIZON_DAYS)
        cuts = pd.date_range(TRAIN_START, last_train_cutoff, freq=f"{TRAIN_STEP_DAYS}D")
        Xs, ys = [], []
        for c in cuts:
            t, X = make_dataset(series, static, c, train_ids, cat_maps, sample=True)
            Xs.append(X); ys.append((t.km - t.last_km).clip(lower=0))
        X_tr, y_tr = pd.concat(Xs), pd.concat(ys)
        print(f"\n=== Stichtag {E}: Training auf {len(cuts)} Stichtagen, {len(X_tr)} Zeilen ===")
        model = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.08, max_iter=400,
                                              max_leaf_nodes=48, min_samples_leaf=100,
                                              early_stopping=True, random_state=0)
        model.fit(X_tr, y_tr)
        print(f"Iterationen nach Early Stopping: {model.n_iter_}")

        t, X = make_dataset(series, static, E, hold_ids, cat_maps)
        t["m_gboost"] = t.last_km + model.predict(X).clip(min=0)
        preds = [c for c in t.columns if c.startswith("b_") or c == "m_gboost"]
        res = by_horizon(t, preds)
        res.insert(0, "cutoff", E)
        out.append(res)
        print(f"Bewertung auf {t.vhcl_anon_id.nunique()} Holdout-Fahrzeugen, {len(t)} Zielmessungen")
        show = res[res.horizon.isin(["all", "0-14", "31-60", "91-10000"]) & res.model.isin(["b_last", "b_segment", "b_rate_all", "m_gboost"])]
        print(show.round(1).to_string(index=False))

    results = RESULTS
    results.mkdir(exist_ok=True)
    pd.concat(out).to_csv(results / "model_metrics.csv", index=False)
    print("\nGespeichert: results/model_metrics.csv")


if __name__ == "__main__":
    main()
