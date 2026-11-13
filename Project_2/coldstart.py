"""Neustart-Fall (Cold Start): Fahrzeuge OHNE bekannte Kilometerhistorie.

In test_data.parquet sind das 149 Neufahrzeuge (Baujahr 2026/27), deren erste Zeile nach dem 01.07.2026 liegt und fuer die
JEDE Zeile ohne Kilometerstand ist. Regel: Start bei einem typischen Neufahrzeug-Stand (Median ca. 12 km) plus die
mediane Wachstumskurve neuer Fahrzeuge nach Alter (Tage seit der ersten Zeile).

Warum eine Kurve statt "Start 0 + Flotten-Rate x Tage"? Im Pseudo-Test (evaluate_coldstart.py) war die Kurve an drei Stichtagen
gleich gut oder besser (siehe review/COLDSTART.md).
"""
import numpy as np
import pandas as pd

HORIZON_DAYS = 133
SERIES_START = "2022-06-01"      # Fahrzeuge, deren erste Messung spaeter liegt, gelten im Sample als "neu"
FIT_START = "2022-07-01"


def new_vehicle_ids(series: pd.DataFrame) -> pd.Index:
    """Fahrzeuge, die im Sample erst nach SERIES_START auftauchen (also echte Neuzugaenge, keine abgeschnittenen Altfahrzeuge)."""
    first = series.groupby("vhcl_anon_id").date.min()
    return first[first > SERIES_START].index


def fit_age_curve(series: pd.DataFrame, vehicle_ids, cutoff, horizon: int = HORIZON_DAYS):
    """Median-Zuwachs seit der ersten Messung je Alter in Tagen. Nutzt nur Messungen <= cutoff und nur Fahrzeuge,
    deren erste Messung mindestens `horizon` Tage vor dem cutoff liegt (volle Kurve, kein Blick in die Zukunft).

    Gibt (curve: Series Alter 0..horizon -> Zuwachs in km, k0: typischer Startstand) zurueck.
    """
    cutoff = pd.Timestamp(cutoff)
    new_ids = new_vehicle_ids(series)
    s = series[series.vhcl_anon_id.isin(vehicle_ids) & series.vhcl_anon_id.isin(new_ids) & (series.date <= cutoff)]
    first = s.groupby("vhcl_anon_id").agg(d0=("date", "first"), k0=("km", "first"))
    ok = first[(first.d0 >= FIT_START) & (first.d0 <= cutoff - pd.Timedelta(days=horizon))]
    m = s.merge(ok, left_on="vhcl_anon_id", right_index=True)
    m["age"] = (m.date - m.d0).dt.days
    m = m[m.age <= horizon]
    grow = (m.km - m.k0).groupby(m.age).median()
    curve = grow.reindex(range(horizon + 1)).interpolate(limit_direction="both").cummax()  # Stand faellt nie
    return curve, float(ok.k0.median())


def predict_from_age(age, curve: pd.Series, k0: float) -> np.ndarray:
    a = np.clip(np.asarray(age, dtype=int), 0, int(curve.index.max()))
    return k0 + curve.reindex(a).to_numpy()


def predict_test_coldstart(test_df: pd.DataFrame, curve: pd.Series, k0: float) -> pd.Series:
    """Vorhersagen fuer alle Zeilen von Fahrzeugen, die in der Testdatei GAR KEINEN bekannten Stand haben.
    Index = Index von test_df (damit man es direkt zurueckschreiben kann)."""
    date = pd.to_datetime(test_df["vhcl_fct_taken_datm"]).dt.normalize()
    known = set(test_df.loc[test_df.vhcl_current_mileage.notna(), "vhcl_anon_id"])
    cold = ~test_df.vhcl_anon_id.isin(known)
    c = test_df[cold].assign(date=date[cold])
    age = (c.date - c.groupby("vhcl_anon_id").date.transform("min")).dt.days
    return pd.Series(predict_from_age(age, curve, k0), index=c.index, name="pred_km")
