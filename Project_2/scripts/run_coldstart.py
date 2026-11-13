"""Pseudo-Test fuer den Neustart-Fall: simulierte Neufahrzeuge (erste Messung im Pruefzeitraum), nur Holdout-Fahrzeuge.

Verglichen werden drei Regeln:
  zero        : Stand bleibt 0
  fleet_rate  : Start 0 + Segment-Median-Rate (wie b_segment) x Tage seit erster Zeile
  age_curve   : coldstart.py (Startstand + mediane Wachstumskurve nach Alter)

Usage (aus Project_2/): python -m scripts.run_coldstart
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.coldstart import HORIZON_DAYS, fit_age_curve, new_vehicle_ids, predict_from_age
from src.config import RESULTS
from src.data import SAMPLE, daily_series, load_raw, split_vehicles, vehicle_static
from src.baselines import make_task
from src.metrics import summarize

CUTOFFS = ["2025-07-01", "2025-10-01", "2026-02-01"]


def main():
    raw = load_raw(SAMPLE)
    series = daily_series(raw)
    static = vehicle_static(raw)
    train_ids, hold_ids = split_vehicles(series.vhcl_anon_id.unique())
    new_ids = new_vehicle_ids(series)
    first = series.groupby("vhcl_anon_id").date.min().rename("d0")
    s = series.join(first, on="vhcl_anon_id").join(static[["vehicle_level"]], on="vhcl_anon_id")
    s["age"] = (s.date - s.d0).dt.days

    rows = []
    for E in CUTOFFS:
        E = pd.Timestamp(E)
        end = E + pd.Timedelta(days=HORIZON_DAYS)
        curve, k0 = fit_age_curve(series, train_ids, E)

        _, t = make_task(series, E)                      # Flotten-Rate wie b_segment (nur Historie bis E)
        g = t.drop_duplicates("vhcl_anon_id").join(static[["vehicle_level"]], on="vhcl_anon_id")
        seg, glob = g.groupby("vehicle_level").rate_90.median(), g.rate_90.median()

        ev = s[s.vhcl_anon_id.isin(hold_ids) & s.vhcl_anon_id.isin(new_ids) & (s.d0 > E) & (s.d0 <= end) & (s.date <= end)].copy()
        ev["zero"] = 0.0
        ev["fleet_rate"] = ev.vehicle_level.map(seg).fillna(glob) * ev.age
        ev["age_curve"] = predict_from_age(ev.age, curve, k0)
        print(f"\nStichtag {E.date()}: {ev.vhcl_anon_id.nunique()} simulierte Neufahrzeuge, {len(ev)} Zeilen, Startstand {k0:.0f} km")
        for m in ["zero", "fleet_rate", "age_curve"]:
            r = summarize(ev.km, ev[m])
            rows.append({"cutoff": E.date(), "rule": m, **r})
            print(f"  {m:11s} MAE {r['MAE']:7.0f}  MedAE {r['MedAE']:7.0f}  WAPE {r['WAPE_%']:5.1f}%  bias {r['bias']:7.0f}")

    out = RESULTS
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "coldstart_metrics.csv", index=False)


if __name__ == "__main__":
    main()
