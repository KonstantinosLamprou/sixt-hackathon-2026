"""Pseudo-Test + Baselines fuer den Forecast.

Die echte Testdatei: Historie bis 01.07.2026, gesucht sind Staende bis 11.11.2026 (ca. 133 Tage).
Wir bilden das im Sample nach: Stichtag (cutoff) waehlen, nur Daten <= cutoff als Input, Ziel sind die
tatsaechlichen Messungen in (cutoff, cutoff + 133 Tage].

Usage (aus Project_2/): python -m scripts.run_baselines
"""
from pathlib import Path

import pandas as pd

from src.config import RESULTS
from src.data import SAMPLE, daily_series, load_raw, split_vehicles, vehicle_static
from src.metrics import by_horizon

HORIZON_DAYS = 133
CUTOFFS = ["2025-10-01", "2026-02-01"]   # Ziel-Ende muss <= 2026-07-01 liegen
RATE_WINDOWS = (30, 90, 180)
MIN_SPAN_DAYS = 14


def make_task(series: pd.DataFrame, cutoff: str, horizon: int = HORIZON_DAYS):
    """Gibt (history_summary, targets) zurueck. history_summary: letzter bekannter Stand + Raten je Fahrzeug."""
    cutoff = pd.Timestamp(cutoff)
    hist = series[series.date <= cutoff]
    tgt = series[(series.date > cutoff) & (series.date <= cutoff + pd.Timedelta(days=horizon))]

    last = hist.groupby("vhcl_anon_id").agg(last_date=("date", "last"), last_km=("km", "last"),
                                            first_date=("date", "first"), first_km=("km", "first"),
                                            n_hist=("km", "size"))
    last["rate_all"] = _rate(last.first_km, last.last_km, last.first_date, last.last_date)
    for w in RATE_WINDOWS:
        win = hist[hist.date >= cutoff - pd.Timedelta(days=w)].groupby("vhcl_anon_id").agg(
            d0=("date", "first"), k0=("km", "first"), d1=("date", "last"), k1=("km", "last"))
        last[f"rate_{w}"] = _rate(win.k0, win.k1, win.d0, win.d1)
    t = tgt.merge(last, left_on="vhcl_anon_id", right_index=True, how="inner")
    t["horizon"] = (t.date - cutoff).dt.days
    t["gap"] = (t.date - t.last_date).dt.days            # Tage seit der letzten bekannten Messung
    return last, t


def _rate(k0, k1, d0, d1):
    span = (d1 - d0).dt.days
    return ((k1 - k0) / span).where(span >= MIN_SPAN_DAYS)


def add_baselines(t: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    t = t.join(static[["vehicle_level"]], on="vhcl_anon_id")
    g = t.drop_duplicates("vhcl_anon_id")
    # Fallback-Rate fuer Fahrzeuge ohne eigene Rate: Median ihres Segments, sonst global
    seg = g.groupby("vehicle_level")["rate_90"].median()
    glob = g["rate_90"].median()
    t["rate_fb"] = t.vehicle_level.map(seg).fillna(glob)
    t["b_last"] = t.last_km                                                          # nichts aendert sich
    t["b_rate_all"] = t.last_km + t.rate_all.fillna(t.rate_fb) * t.gap              # Durchschnitt ueber alles
    for w in RATE_WINDOWS:
        t[f"b_rate_{w}"] = t.last_km + t[f"rate_{w}"].fillna(t.rate_all).fillna(t.rate_fb) * t.gap
    t["b_segment"] = t.last_km + t.rate_fb * t.gap                                   # nur Segment-Median
    return t


def main():
    raw = load_raw(SAMPLE)
    series = daily_series(raw)
    static = vehicle_static(raw)
    print(f"{series.vhcl_anon_id.nunique()} Fahrzeuge, {len(series)} Tagesmessungen; "
          f"Flags: {series.flag.value_counts().drop('', errors='ignore').to_dict()}")
    _, holdout = split_vehicles(series.vhcl_anon_id.unique())
    out = []
    for c in CUTOFFS:
        _, t = make_task(series, c)
        t = add_baselines(t, static)
        t = t[t.gap > 0]
        preds = [col for col in t.columns if col.startswith("b_")]
        print(f"\n=== Stichtag {c}: {t.vhcl_anon_id.nunique()} Fahrzeuge, {len(t)} Zielmessungen "
              f"(davon Holdout-Fahrzeuge: {t.vhcl_anon_id.isin(holdout).sum()}) ===")
        res = by_horizon(t, preds)
        res.insert(0, "cutoff", c)
        out.append(res)
        print(res[res.horizon.isin(["alle", "0-14", "31-60", "91-10000"])].round(1).to_string(index=False))
    results = RESULTS
    results.mkdir(exist_ok=True)
    pd.concat(out).to_csv(results / "baseline_metrics.csv", index=False)
    print("\nGespeichert: results/baseline_metrics.csv")


if __name__ == "__main__":
    main()
