"""Kette Nowcast -> Forecast.

Idee: Die letzte Messung ist oft veraltet (stale_days). Wir schaetzen zuerst den Stand am Stichtag E (Nowcast), setzen ihn als
neuen Startpunkt (last_km, last_date = E) und lassen den Forecast von dort laufen.

Der Status am Stichtag E ist nicht direkt bekannt (es gibt nicht an jedem Tag eine Zeile). Wir uebertragen den Status der letzten
Zeile: Miete laeuft weiter, solange das geplante Rueckgabedatum >= E ist; sonst gilt das Fahrzeug als nicht vermietet.

Vergleich auf denselben Holdout-Zeilen: b_segment, m_forecast (wie model.py), m_chain_hist (Nowcast ohne Status),
m_chain_status (Nowcast mit Status). Zusaetzlich getrennt nach Alter der letzten Messung.
Usage (aus Project_2/): python -m experiments.chain
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import src.forecast as fc
import src.nowcast as nc
from src.config import RESULTS
from src.data import SAMPLE, daily_series, load_raw, split_vehicles, vehicle_static
from src.baselines import CUTOFFS, HORIZON_DAYS, add_baselines
from src.metrics import summarize

VEH_COLS = ["vhcl_anon_id", "last_date", "last_km", "first_date", "n_hist", "rate_all", "rate_30", "rate_90", "rate_180"]
SUBSETS = {"alle": lambda s: s >= 0, "stale<=7": lambda s: s <= 7, "stale>7": lambda s: s > 7, "stale>14": lambda s: s > 14}


def train_forecast(series, static, cat_maps, train_ids, E):
    cuts = pd.date_range(fc.TRAIN_START, pd.Timestamp(E) - pd.Timedelta(days=HORIZON_DAYS), freq=f"{fc.TRAIN_STEP_DAYS}D")
    parts = [fc.make_dataset(series, static, c, train_ids, cat_maps, sample=True) for c in cuts]
    y = pd.concat([(t.km - t.last_km).clip(lower=0) for t, _ in parts])
    m = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.08, max_iter=400, max_leaf_nodes=48,
                                      min_samples_leaf=100, early_stopping=True, random_state=0)
    return m.fit(pd.concat([p[1] for p in parts]), y)


def train_nowcast(series, static, cat_maps, train_ids, E):
    cuts = pd.date_range(nc.TRAIN_START, pd.Timestamp(E) - pd.Timedelta(days=nc.NOWCAST_DAYS), freq=f"{nc.TRAIN_STEP_DAYS}D")
    parts = [nc.make_dataset(series, static, c, train_ids, cat_maps, sample=True) for c in cuts]
    y = pd.concat([(t.km - t.last_km).clip(lower=0) for t, _, _ in parts])
    Xb, Xs = pd.concat([p[1] for p in parts]), pd.concat([p[2] for p in parts])
    return nc.new_model().fit(Xb, y), nc.new_model().fit(Xs, y)


def carried_status(series, t_veh, E):
    """Status der letzten Zeile, auf den Stichtag E fortgeschrieben."""
    last = series[series.date <= E].groupby("vhcl_anon_id").tail(1).set_index("vhcl_anon_id")
    s = t_veh[["vhcl_anon_id", "gap"]].join(last[nc.STATUS_FEATURES], on="vhcl_anon_id")
    still = (s.days_to_return - s.gap >= 0) & (s.on_rent == 1)
    s["days_since_handover"] = (s.days_since_handover + s.gap).where(still)
    s["days_to_return"] = (s.days_to_return - s.gap).where(still)
    s["rental_len"] = s.rental_len.where(still)
    s["on_rent"] = still.astype(int)
    s["n_rows_day"] = 1
    return s.set_index(t_veh.index)[nc.STATUS_FEATURES]


def main():
    raw = load_raw(SAMPLE)
    series = daily_series(raw).merge(nc.daily_status(raw), on=["vhcl_anon_id", "date"], how="left")
    static = vehicle_static(raw)
    cat_maps = {c: {v: i for i, v in enumerate(sorted(static[c].dropna().unique()))} for c in fc.CAT_COLS}
    train_ids, hold_ids = split_vehicles(series.vhcl_anon_id.unique())

    rows = []
    for E in ["2025-07-01"] + [c for c in CUTOFFS]:
        Et = pd.Timestamp(E)
        F = train_forecast(series, static, cat_maps, train_ids, E)
        N_hist, N_stat = train_nowcast(series, static, cat_maps, train_ids, E)

        t, X = fc.make_dataset(series, static, E, hold_ids, cat_maps)
        t["stale"] = (Et - t.last_date).dt.days
        t["m_forecast"] = t.last_km + F.predict(X).clip(min=0)

        v = t.drop_duplicates("vhcl_anon_id")[VEH_COLS].copy()
        v["date"], v["km"], v["horizon"] = Et, np.nan, 0
        v["gap"] = (Et - v.last_date).dt.days
        v = add_baselines(v, static)
        Xv = fc.build_features(v, static, Et, cat_maps)
        Xv_stat = Xv.join(carried_status(series, v, Et))
        nowc = {"m_chain_hist": v.last_km + N_hist.predict(Xv).clip(min=0),
                "m_chain_status": v.last_km + N_stat.predict(Xv_stat).clip(min=0)}
        for name, n_e in nowc.items():
            n_e = n_e.where(v.gap > 0, v.last_km)                                    # Messung genau am Stichtag: kein Nowcast noetig
            tc = t[[c for c in t.columns if c not in ("vehicle_level", "rate_fb") and not c.startswith("b_")]].copy()
            tc["last_km"] = tc.vhcl_anon_id.map(pd.Series(n_e.values, index=v.vhcl_anon_id))
            tc["last_date"], tc["gap"] = Et, tc["horizon"]
            tc = add_baselines(tc, static)
            Xc = fc.build_features(tc, static, Et, cat_maps)
            t[name] = tc.last_km + F.predict(Xc).clip(min=0)

        print(f"\n=== Stichtag {E}: {t.vhcl_anon_id.nunique()} Holdout-Fahrzeuge, {len(t)} Zeilen ===")
        for sub, f in SUBSETS.items():
            s = t[f(t.stale)]
            for m in ["b_segment", "m_forecast", "m_chain_hist", "m_chain_status"]:
                r = summarize(s.km, s[m])
                rows.append({"cutoff": E, "subset": sub, "model": m, **r})
        res = pd.DataFrame(rows)
        print(res[res.cutoff == E].pivot(index="subset", columns="model", values="MAE").loc[list(SUBSETS)].round(0).to_string())
        print("n:", {k: int(v) for k, v in res[(res.cutoff == E) & (res.model == "m_forecast")].set_index("subset").n.items()})

    out = RESULTS
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "chain_metrics.csv", index=False)


if __name__ == "__main__":
    main()
