"""Vorhersage fuer die leeren Zeilen von test_data.parquet (Teil A, Forecast).

- Fahrzeuge MIT bekannter Historie: getuntes Gradient-Boosting-Modell (Konfiguration aus results/best_forecast_params.json),
  trainiert NUR auf den Sample-Fahrzeugen (Stichtag = 2026-07-01). Der Rueckwaertstest auf den Testfahrzeugen (src/backtest.py)
  zeigte, dass die fruehe Historie der Testfahrzeuge im Training nichts bringt (MAE 1.400 statt 1.396 km).
- Fahrzeuge OHNE Historie (149 Neufahrzeuge): Alterskurve aus src/coldstart.py.

Start (aus Project_2/):  python -m scripts.run_predict_test
Ausgabe: submission/predictions_forecast.csv, models/forecast_final_2026-07-01.joblib
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

import src.forecast as fc
from src.baselines import HORIZON_DAYS, add_baselines, make_task
from src.coldstart import fit_age_curve, predict_test_coldstart
from src.config import MODELS, RESULTS, ROOT
from src.data import SAMPLE, TEST, daily_series, load_raw, vehicle_static

CUTOFF = "2026-07-01"


def main():
    cfg = json.load(open(RESULTS / "best_forecast_params.json"))["cfg"]
    raw_s, raw_t = load_raw(SAMPLE), load_raw(TEST)
    ser_s, ser_t = daily_series(raw_s), daily_series(raw_t)             # Testreihe: nur Zeilen mit bekanntem Stand
    series = pd.concat([ser_s, ser_t], ignore_index=True)
    static = pd.concat([vehicle_static(raw_s), vehicle_static(raw_t)])
    static = static[~static.index.duplicated()]
    cat_maps = {c: {v: i for i, v in enumerate(sorted(static[c].dropna().unique()))} for c in fc.CAT_COLS}
    E = pd.Timestamp(CUTOFF)

    # ---- Training auf allen Fahrzeugen (Sample + bekannte Testhistorie) -----------------------------------------------
    cuts = pd.date_range(fc.TRAIN_START, E - pd.Timedelta(days=HORIZON_DAYS), freq=f"{fc.TRAIN_STEP_DAYS}D")
    ids = ser_s.vhcl_anon_id.unique()                                   # nur Sample-Fahrzeuge (siehe Rueckwaertstest)
    parts = [fc.make_dataset(series, static, c, ids, cat_maps, sample=True) for c in cuts]
    X_tr = pd.concat([p[1] for p in parts])
    y_tr = pd.concat([(t.km - t.last_km).clip(lower=0) for t, _ in parts])
    print(f"Training: {len(cuts)} Stichtage, {len(X_tr)} Zeilen, {len(ids)} Fahrzeuge (nur Sample)", flush=True)
    model = HistGradientBoostingRegressor(loss="absolute_error", early_stopping=False, random_state=0, **cfg).fit(X_tr, y_tr)
    joblib.dump({"model": model, "cfg": cfg, "columns": list(X_tr.columns), "cutoff": CUTOFF}, MODELS / "forecast_final_2026-07-01.joblib",
                compress=3)

    # ---- Vorhersage fuer Fahrzeuge mit Historie -------------------------------------------------------------------------
    empty = raw_t[raw_t.vhcl_current_mileage.isna()]
    tg = empty[["vhcl_anon_id", "date"]].drop_duplicates().assign(km=np.nan, flag="")
    ser_pred = pd.concat([ser_t[["vhcl_anon_id", "date", "km", "flag"]], tg], ignore_index=True)
    _, t = make_task(ser_pred, CUTOFF)
    t = add_baselines(t[t.gap > 0], static)
    X = fc.build_features(t, static, E, cat_maps)
    t["pred"] = t.last_km + model.predict(X).clip(min=0)
    pred_model = t[["vhcl_anon_id", "date", "pred"]].drop_duplicates(["vhcl_anon_id", "date"]).sort_values(["vhcl_anon_id", "date"])
    # Ein Kilometerstand faellt nie: Vorhersagen je Fahrzeug ueber die Zeit monoton machen (isotone Regression)
    pred_model["pred"] = pred_model.groupby("vhcl_anon_id").pred.transform(
        lambda s: IsotonicRegression().fit_transform(np.arange(len(s)), s.to_numpy()))

    out = empty[["vhcl_anon_id", "vhcl_fct_taken_datm", "date"]].merge(pred_model, on=["vhcl_anon_id", "date"], how="left")
    out.index = empty.index
    out["method"] = np.where(out.pred.notna(), "model", "")

    # ---- Neufahrzeuge ohne Historie -------------------------------------------------------------------------------------
    curve, k0 = fit_age_curve(ser_s, ser_s.vhcl_anon_id.unique(), CUTOFF)
    cold = predict_test_coldstart(raw_t, curve, k0)
    out.loc[cold.index, "pred"] = cold
    out.loc[cold.index, "method"] = "cold_start"

    missing = out.pred.isna()
    print(f"Zeilen: {len(out)}, model: {(out.method == 'model').sum()}, cold_start: {(out.method == 'cold_start').sum()}, "
          f"ohne Vorhersage: {missing.sum()}")
    if missing.any():     # Rueckfall: letzter bekannter Stand
        lastk = ser_t.groupby("vhcl_anon_id").km.last()
        out.loc[missing, "pred"] = out.loc[missing, "vhcl_anon_id"].map(lastk)
        out.loc[missing, "method"] = "fallback_last"
        print("Rueckfall letzter Stand fuer", int(missing.sum()), "Zeilen; noch NaN:", int(out.pred.isna().sum()))

    res = out.rename(columns={"pred": "predicted_mileage"})[["vhcl_anon_id", "vhcl_fct_taken_datm", "predicted_mileage", "method"]]
    res["predicted_mileage"] = res.predicted_mileage.round(1)
    (ROOT / "submission").mkdir(exist_ok=True)
    res.to_csv(ROOT / "submission" / "predictions_forecast.csv", index=False)
    print(res.groupby("method").predicted_mileage.describe().round(0).to_string())
    print("Gespeichert: submission/predictions_forecast.csv")


if __name__ == "__main__":
    main()
