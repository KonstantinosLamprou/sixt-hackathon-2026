"""Laden und Bereinigen der Project-2-Daten.

Kernidee: Pro Fahrzeug und Tag gibt es oft mehrere Zeilen (OnRent/NotOnRent). Wir fassen sie zu EINER
Messung pro Fahrzeug-Tag zusammen, markieren Ausreisser/Rueckwaertssprünge (Teil B: Regel-Flags) und
erzwingen eine monoton steigende Kilometerreihe (Teil A: sauberes Trainings-/Baseline-Signal).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR

HERE = DATA_DIR.parent
DATA = DATA_DIR
SAMPLE = DATA / "data_30_percent_sample.parquet"
TEST = DATA / "test_data.parquet"

STATIC_COLS = ["acriss_code_anonymized", "vehicle_level", "vehicle_type", "model_year", "driven_wheels",
               "power_ps", "cubic_capacity_liter", "vhcl_max_mileage"]
DROP_COLS = ["sixt_predicted_mileage_interp", "sixt_error"]  # in den Rohdaten komplett leer

SPIKE_KM = 10_000      # Abweichung vom lokalen Median, ab der eine Messung als Ausreisser gilt
DROP_TOL_KM = 50       # Rueckgang gegenueber dem bisherigen Maximum, ab dem eine Messung als "faellt" gilt


def load_raw(path: Path = SAMPLE) -> pd.DataFrame:
    df = pd.read_parquet(path).drop(columns=DROP_COLS, errors="ignore")
    df["date"] = pd.to_datetime(df["vhcl_fct_taken_datm"]).dt.normalize()
    return df


def vehicle_static(df: pd.DataFrame) -> pd.DataFrame:
    """Eine Zeile pro Fahrzeug mit den (fast) konstanten Attributen."""
    cols = [c for c in STATIC_COLS if c in df]
    return df.groupby("vhcl_anon_id")[cols].last()


def daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Eine Messung pro Fahrzeug-Tag. Spalten: vhcl_anon_id, date, raw_km, km (bereinigt, monoton), flag.

    flag: '' = ok, 'spike' = weicht stark vom lokalen Median ab (wird fuer km ignoriert),
          'decrease' = faellt unter das bisherige Maximum (km wird auf das Maximum gehoben).
    """
    d = df.dropna(subset=["vhcl_current_mileage"])
    # max pro Tag: NotOnRent-Zeilen tragen oft den Stand NACH der Fahrt, der Tageswert ist der hoechste.
    d = (d.groupby(["vhcl_anon_id", "date"], sort=True)["vhcl_current_mileage"].max()
         .rename("raw_km").reset_index())
    g = d.groupby("vhcl_anon_id", sort=False)["raw_km"]
    med = g.rolling(7, center=True, min_periods=1).median().reset_index(level=0, drop=True)
    spike = (d["raw_km"] - med).abs() > SPIKE_KM
    km = d["raw_km"].where(~spike)
    km = km.groupby(d["vhcl_anon_id"], sort=False).cummax()          # monoton; Spikes (NaN) ueberspringen
    km = km.groupby(d["vhcl_anon_id"], sort=False).ffill()
    decrease = ~spike & (d["raw_km"] < km - DROP_TOL_KM)
    d["km"] = km
    d["flag"] = np.where(spike, "spike", np.where(decrease, "decrease", ""))
    return d.dropna(subset=["km"]).reset_index(drop=True)


def split_vehicles(ids, holdout_frac: float = 0.2, seed: int = 0):
    """Deterministischer Fahrzeug-Split (kein Zeilen-Split!). Gibt (train_ids, holdout_ids) zurueck."""
    ids = np.array(sorted(set(ids)))
    rng = np.random.default_rng(seed)
    mask = rng.random(len(ids)) < holdout_frac
    return ids[~mask], ids[mask]
