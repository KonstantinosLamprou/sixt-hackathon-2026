"""Fehlermetriken fuer die Mileage-Vorhersage."""
import numpy as np
import pandas as pd

BUCKETS = [(0, 14), (15, 30), (31, 60), (61, 90), (91, 10_000)]


def summarize(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    return {"n": len(y), "MAE": np.abs(err).mean(), "MedAE": np.median(np.abs(err)),
            "WAPE_%": 100 * np.abs(err).sum() / np.abs(y).sum(), "bias": err.mean()}


def by_horizon(frame: pd.DataFrame, pred_cols, y_col="km", h_col="horizon", buckets=None) -> pd.DataFrame:
    """frame: Zeilen mit Zielwert, Horizont in Tagen und einer Spalte je Modell. Tabelle: Modell x Horizont-Bucket."""
    rows = []
    for name in pred_cols:
        sub = frame.dropna(subset=[name])
        for lo, hi in [(0, 10_000)] + (buckets or BUCKETS):
            s = sub[(sub[h_col] >= lo) & (sub[h_col] <= hi)]
            if len(s):
                rows.append({"model": name, "horizon": "alle" if hi == 10_000 and lo == 0 else f"{lo}-{hi}",
                             **summarize(s[y_col], s[name])})
    return pd.DataFrame(rows)
