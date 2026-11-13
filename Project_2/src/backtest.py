"""Rueckwaertstest auf der TESTFLOTTE (echte Testfahrzeuge, bekannter Zeitraum).

Die wahren Werte ab 02.07.2026 kennen wir nicht. Aber die Testdatei enthaelt fuer ihre Fahrzeuge bekannte Staende bis 01.07.2026.
Wir tun so, als wuessten wir nur Daten bis 18.02.2026 (= 01.07.2026 minus 133 Tage), sagen 133 Tage voraus und vergleichen mit den
tatsaechlich bekannten Staenden bis 01.07.2026. Das ist derselbe Aufbau wie die echte Aufgabe, auf denselben Fahrzeugen.

Trainiert wird mit Zielen, die spaetestens am 18.02.2026 enden (nichts aus dem Pruefzeitraum). Verglichen werden:
  b_last, b_segment          : einfache Regeln
  m_sample                   : Modell, trainiert NUR auf Sample-Fahrzeugen (Testfahrzeuge nie gesehen)
  m_sample+test              : Modell, trainiert auf Sample + frueher Historie der Testfahrzeuge
  jeweils mit / ohne monotone Glaettung pro Fahrzeug (isotone Regression)
Start (aus Project_2/):  python -m scripts.run_backtest_test
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

import src.forecast as fc
from src.baselines import HORIZON_DAYS
from src.config import RESULTS
from src.data import SAMPLE, TEST, daily_series, load_raw, vehicle_static
from src.metrics import by_horizon

CUT = pd.Timestamp("2026-02-18")


def iso(t, col):
    t = t.sort_values(["vhcl_anon_id", "date"])
    return t.groupby("vhcl_anon_id")[col].transform(lambda s: IsotonicRegression().fit_transform(np.arange(len(s)), s.to_numpy()))


def train(series, static, cat_maps, ids, cfg):
    cuts = pd.date_range(fc.TRAIN_START, CUT - pd.Timedelta(days=HORIZON_DAYS), freq=f"{fc.TRAIN_STEP_DAYS}D")
    parts = [fc.make_dataset(series, static, c, ids, cat_maps, sample=True) for c in cuts]
    X, y = pd.concat([p[1] for p in parts]), pd.concat([(p[0].km - p[0].last_km).clip(lower=0) for p in parts])
    print(f"  Training: {len(cuts)} Stichtage, {len(X)} Zeilen", flush=True)
    return HistGradientBoostingRegressor(loss="absolute_error", early_stopping=False, random_state=0, **cfg).fit(X, y)


def main():
    cfg = json.load(open(RESULTS / "best_forecast_params.json"))["cfg"]
    raw_s, raw_t = load_raw(SAMPLE), load_raw(TEST)
    ser_s, ser_t = daily_series(raw_s), daily_series(raw_t)
    static = pd.concat([vehicle_static(raw_s), vehicle_static(raw_t)])
    static = static[~static.index.duplicated()]
    cat_maps = {c: {v: i for i, v in enumerate(sorted(static[c].dropna().unique()))} for c in fc.CAT_COLS}
    series = pd.concat([ser_s, ser_t], ignore_index=True)
    sample_ids, test_ids = ser_s.vhcl_anon_id.unique(), ser_t.vhcl_anon_id.unique()

    print("Modell A: nur Sample-Fahrzeuge")
    m_a = train(series, static, cat_maps, sample_ids, cfg)
    print("Modell B: Sample + Historie der Testfahrzeuge")
    m_b = train(series, static, cat_maps, np.concatenate([sample_ids, test_ids]), cfg)

    t, X = fc.make_dataset(series, static, CUT, test_ids, cat_maps)       # Pruefung nur auf Testfahrzeugen
    t["m_sample"] = t.last_km + m_a.predict(X).clip(min=0)
    t["m_sample+test"] = t.last_km + m_b.predict(X).clip(min=0)
    t["m_sample_iso"], t["m_sample+test_iso"] = iso(t, "m_sample"), iso(t, "m_sample+test")
    t = t.sort_values(["vhcl_anon_id", "date"])
    t["m_sample_iso"], t["m_sample+test_iso"] = iso(t, "m_sample"), iso(t, "m_sample+test")
    preds = ["b_last", "b_segment", "m_sample", "m_sample_iso", "m_sample+test", "m_sample+test_iso"]
    res = by_horizon(t, preds)
    res.to_csv(RESULTS / "backtest_test_fleet.csv", index=False)
    print(f"\nRueckwaertstest: {t.vhcl_anon_id.nunique()} Testfahrzeuge, {len(t)} Messungen, Stichtag {CUT.date()}, bis 01.07.2026")
    print(res[res.horizon == "all"].round(1).to_string(index=False))
    print(res[res.horizon.isin(["0-14", "31-60", "91-10000"])].pivot(index="model", columns="horizon", values="MAE").round(0).to_string())


if __name__ == "__main__":
    main()
