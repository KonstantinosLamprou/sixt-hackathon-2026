"""Gemeinsame Abgabedatei: eine Zeile je Zeile von test_data.parquet (gleiche Reihenfolge), mit row_id.

Spalten: row_id, vhcl_anon_id, vhcl_fct_taken_datm, vhcl_current_mileage (Original), forecast_mileage + forecast_method (nur leere
Zeilen), mileage_error_flag + flag_score + flag_reason (nur Zeilen mit gemeldetem Stand).
Voraussetzung: submission/predictions_forecast.csv (run_predict_test) und data/interim/partb_flags_all_rows.parquet (run_partb).
Start (aus Project_2/):  python -m scripts.run_export
"""
import numpy as np
import pandas as pd

from src.config import INTERIM, ROOT
from src.data import TEST, load_raw


def main():
    raw = load_raw(TEST)
    n = len(raw)
    out = pd.DataFrame({"row_id": np.arange(n), "vhcl_anon_id": raw.vhcl_anon_id.values,
                        "vhcl_fct_taken_datm": raw.vhcl_fct_taken_datm.values,
                        "vhcl_current_mileage": raw.vhcl_current_mileage.values})
    empty = raw.vhcl_current_mileage.isna().values
    known = ~empty

    fc = pd.read_csv(ROOT / "submission" / "predictions_forecast.csv")
    assert len(fc) == empty.sum() and (fc.vhcl_anon_id.values == raw.vhcl_anon_id.values[empty]).all(), "Forecast-Reihenfolge stimmt nicht"
    out["forecast_mileage"], out["forecast_method"] = np.nan, ""
    out.loc[empty, "forecast_mileage"] = fc.predicted_mileage.values
    out.loc[empty, "forecast_method"] = fc.method.values

    fl = pd.read_parquet(INTERIM / "partb_flags_all_rows.parquet")
    assert len(fl) == known.sum() and (fl.vhcl_anon_id.values == raw.vhcl_anon_id.values[known]).all(), "Flag-Reihenfolge stimmt nicht"
    out["mileage_error_flag"] = pd.array([pd.NA] * n, dtype="Int8")
    out["flag_score"], out["flag_reason"] = np.nan, ""
    out.loc[known, "mileage_error_flag"] = fl.flag.values.astype("int8")
    out.loc[known, "flag_score"] = fl.score.round(2).values
    out.loc[known, "flag_reason"] = fl.reason.values

    path = ROOT / "submission" / "predictions.parquet"
    out.to_parquet(path, index=False, compression="zstd")
    print(f"{n} Zeilen -> {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"forecast gefuellt: {int(out.forecast_mileage.notna().sum())} (leere Zeilen: {int(empty.sum())}), "
          f"flag gesetzt: {int(out.mileage_error_flag.notna().sum())}, geflaggt: {int((out.mileage_error_flag == 1).sum())}")


if __name__ == "__main__":
    main()
