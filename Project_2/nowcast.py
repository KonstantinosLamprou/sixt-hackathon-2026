"""Nowcast (Teil A): aktueller Kilometerstand aus letztem bekannten Stand + Status des Zieltages.

Unterschied zum Forecast (model.py): Das Ziel liegt hoechstens 28 Tage nach dem letzten Wissensstand (so lang kann die Meldung
verspaetet sein) und wir kennen den STATUS am Zieltag: laeuft eine Miete, welches Produkt, Einweg, seit wann, bis wann
geplant, Reparatur, uti_plan.

Leakage-Check: rntl_return_date liegt in 100 % der OnRent-Zeilen am Messtag oder danach (31 % in der Zukunft). Es ist also das
zum Zeitpunkt der Zeile bekannte geplante Rueckgabedatum und darf als Feature genutzt werden.

Es werden zwei Modelle auf denselben Zeilen verglichen: ohne Status-Features (nur Historie) und mit Status-Features.
Usage: python nowcast.py
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from data import SAMPLE, daily_series, load_raw, split_vehicles, vehicle_static
from evaluate import add_baselines, make_task
from metrics import by_horizon
from model import CAT_COLS, build_features

NOWCAST_DAYS = 28                  # maximale Meldeverzoegerung
CUTOFFS = ["2025-07-01", "2025-10-01", "2026-02-01"]
TRAIN_START = "2022-07-01"
TRAIN_STEP_DAYS = 14
ROWS_PER_VEHICLE = 4
BUCKETS = [(0, 7), (8, 14), (15, 28)]
STATUS_FEATURES = ["on_rent", "longterm", "one_way", "repair", "status_code", "uti_plan",
                   "days_since_handover", "days_to_return", "rental_len", "n_rows_day"]


def daily_status(raw: pd.DataFrame) -> pd.DataFrame:
    """Status je Fahrzeug-Tag (mehrere Zeilen am Tag werden zusammengefasst)."""
    r = raw[["vhcl_anon_id", "date", "on_rent_status", "rental_product", "one_way_flg", "repair_status",
             "status_code_anonymized", "uti_plan", "rntl_handover_date", "rntl_return_date"]].copy()
    r["on_rent"] = r.on_rent_status == "OnRent"
    r["longterm"] = r.rental_product == "LongTerm"
    r["one_way"] = r.one_way_flg == "OneWay"
    r["repair"] = r.repair_status == "OnRepair"
    r["h"] = pd.to_datetime(r.rntl_handover_date)
    r["rt"] = pd.to_datetime(r.rntl_return_date)
    s = r.groupby(["vhcl_anon_id", "date"]).agg(
        on_rent=("on_rent", "max"), longterm=("longterm", "max"), one_way=("one_way", "max"),
        repair=("repair", "max"), status_code=("status_code_anonymized", "max"), uti_plan=("uti_plan", "mean"),
        h=("h", "max"), rt=("rt", "max"), n_rows_day=("on_rent", "size")).reset_index()
    for c in ["on_rent", "longterm", "one_way", "repair"]:
        s[c] = s[c].astype(int)
    s["days_since_handover"] = (s.date - s.h).dt.days
    s["days_to_return"] = (s.rt - s.date).dt.days
    s["rental_len"] = (s.rt - s.h).dt.days
    return s.drop(columns=["h", "rt"])


def make_dataset(series, static, cutoff, vehicles, cat_maps, sample=False, seed=0):
    _, t = make_task(series, cutoff, horizon=NOWCAST_DAYS)
    t = t[t.vhcl_anon_id.isin(vehicles) & (t.gap > 0)]
    t = add_baselines(t, static)
    if sample:
        t = t.sample(frac=1, random_state=seed).groupby("vhcl_anon_id").head(ROWS_PER_VEHICLE)
    X = build_features(t, static, pd.Timestamp(cutoff), cat_maps)
    return t, X, X.join(t[STATUS_FEATURES])


def new_model():
    return HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.08, max_iter=300, max_leaf_nodes=48,
                                         min_samples_leaf=100, early_stopping=True, random_state=0)


def main():
    raw = load_raw(SAMPLE)
    series = daily_series(raw).merge(daily_status(raw), on=["vhcl_anon_id", "date"], how="left")
    static = vehicle_static(raw)
    cat_maps = {c: {v: i for i, v in enumerate(sorted(static[c].dropna().unique()))} for c in CAT_COLS}
    train_ids, hold_ids = split_vehicles(series.vhcl_anon_id.unique())

    out = []
    for E in CUTOFFS:
        cuts = pd.date_range(TRAIN_START, pd.Timestamp(E) - pd.Timedelta(days=NOWCAST_DAYS), freq=f"{TRAIN_STEP_DAYS}D")
        parts = [make_dataset(series, static, c, train_ids, cat_maps, sample=True) for c in cuts]
        y = pd.concat([(t.km - t.last_km).clip(lower=0) for t, _, _ in parts])
        X_base, X_stat = pd.concat([p[1] for p in parts]), pd.concat([p[2] for p in parts])
        print(f"\n=== Stichtag {E}: Training auf {len(cuts)} Stichtagen, {len(y)} Zeilen ===")
        m_base, m_stat = new_model().fit(X_base, y), new_model().fit(X_stat, y)

        t, Xb, Xs = make_dataset(series, static, E, hold_ids, cat_maps)
        t["m_nc_history"] = t.last_km + m_base.predict(Xb).clip(min=0)
        t["m_nc_status"] = t.last_km + m_stat.predict(Xs).clip(min=0)
        preds = ["b_last", "b_rate_all", "b_segment", "m_nc_history", "m_nc_status"]
        res = by_horizon(t, preds, buckets=BUCKETS)
        res.insert(0, "cutoff", E)
        out.append(res)
        print(f"Bewertung: {t.vhcl_anon_id.nunique()} Holdout-Fahrzeuge, {len(t)} Zielmessungen; "
              f"Anteil Ziele mit unveraendertem Stand: {(t.km == t.last_km).mean():.0%}")
        print(res.round(1).to_string(index=False))

    results = Path(__file__).parent / "results"
    results.mkdir(exist_ok=True)
    pd.concat(out).to_csv(results / "nowcast_metrics.csv", index=False)


if __name__ == "__main__":
    main()
