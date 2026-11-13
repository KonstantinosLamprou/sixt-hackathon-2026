# Abgabedateien Project 2

## `predictions.parquet` (Hauptdatei)
Eine Zeile je Zeile von `data/test_data.parquet`, **gleiche Reihenfolge** (656.601 Zeilen). Zum Zuordnen `row_id` benutzen (= Position in `test_data.parquet`, bei 0 beginnend); `vhcl_anon_id` + `vhcl_fct_taken_datm` sind nicht eindeutig, weil es mehrere Zeilen am selben Tag gibt.

| Spalte | Inhalt | Gefüllt für |
|---|---|---|
| `row_id` | Position in `test_data.parquet` | alle Zeilen |
| `vhcl_anon_id`, `vhcl_fct_taken_datm` | Schlüssel (Original) | alle Zeilen |
| `vhcl_current_mileage` | gemeldeter Stand (Original) | 493.554 Zeilen (bis 01.07.2026) |
| `forecast_mileage` | **vorhergesagter Kilometerstand** (Teil A, Forecast) | die 163.047 leeren Zeilen (02.07. bis 11.11.2026) |
| `forecast_method` | `model` (Gradient Boosting, 159.302 Zeilen) oder `cold_start` (Alterskurve für 149 Neufahrzeuge ohne Historie, 3.745 Zeilen) | leere Zeilen |
| `mileage_error_flag` | **0/1: Meldung auffällig** (Teil B) | Zeilen mit gemeldetem Stand |
| `flag_score` | Stärke der Auffälligkeit (0 = unauffällig; Regelverstöße sind verdoppelt) | Zeilen mit gemeldetem Stand |
| `flag_reason` | Begründung als Text | geflaggte Zeilen (5.338) |

Ein **Nowcast pro Testzeile** ist nicht enthalten: Die leeren Testzeilen haben keinen Mietstatus, und für die bekannten Zeilen wurde kein zeilenweiser Nowcast gebaut. Das Nowcast-Modell (`models/nowcast_status_2026-07-01.joblib`) und seine Ergebnisse auf zurückgehaltenen Fahrzeugen (`results/nowcast_metrics.csv`, `review/NOWCAST.md`) sind Teil der Abgabe.

Die Vorhersagen sind pro Fahrzeug über die Zeit nicht fallend. Die Forecast-Werte ermitteln wir aus dem Stand bis 01.07.2026.

## Lesehilfen
- `predictions_forecast.csv`: nur die leeren Zeilen (`vhcl_anon_id`, `vhcl_fct_taken_datm`, `predicted_mileage`, `method`).
- `../data/processed/partb_flagged_rows.csv`: nur die geflaggten Zeilen mit Score und Begründung.

## Reproduzieren
```bash
python -m scripts.run_predict_test    # Forecast  -> predictions_forecast.csv, models/forecast_final_2026-07-01.joblib
python -m scripts.run_partb           # Flags     -> data/interim/partb_flags_all_rows.parquet
python -m scripts.run_export          # alles     -> predictions.parquet
```
Details und Ergebnisse: `../SUBMISSION.md`.
