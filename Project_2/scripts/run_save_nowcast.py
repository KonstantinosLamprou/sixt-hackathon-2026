"""Nowcast-Modell (mit Status) auf allen Sample-Fahrzeugen trainieren und speichern.

Stichtage alle 14 Tage bis 01.06.2026 (letztes Ziel endet am 01.07.2026). Start:  python -m scripts.run_save_nowcast
Ausgabe: models/nowcast_status_2026-07-01.joblib
"""
import joblib
import pandas as pd

import src.forecast as fc
import src.nowcast as nc
from src.config import MODELS
from src.data import SAMPLE, daily_series, load_raw, vehicle_static


def main():
    raw = load_raw(SAMPLE)
    series = daily_series(raw).merge(nc.daily_status(raw), on=["vhcl_anon_id", "date"], how="left")
    static = vehicle_static(raw)
    cat_maps = {c: {v: i for i, v in enumerate(sorted(static[c].dropna().unique()))} for c in fc.CAT_COLS}
    ids = series.vhcl_anon_id.unique()
    cuts = pd.date_range(nc.TRAIN_START, pd.Timestamp("2026-07-01") - pd.Timedelta(days=nc.NOWCAST_DAYS), freq=f"{nc.TRAIN_STEP_DAYS}D")
    parts = [nc.make_dataset(series, static, c, ids, cat_maps, sample=True) for c in cuts]
    y = pd.concat([(t.km - t.last_km).clip(lower=0) for t, _, _ in parts])
    X = pd.concat([p[2] for p in parts])
    model = nc.new_model().fit(X, y)
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": model, "columns": list(X.columns), "nowcast_days": nc.NOWCAST_DAYS}, MODELS / "nowcast_status_2026-07-01.joblib", compress=3)
    print(f"Nowcast-Modell gespeichert ({len(cuts)} Stichtage, {len(X)} Zeilen)")


if __name__ == "__main__":
    main()
