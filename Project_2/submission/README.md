# Submission files, Project 2

## `predictions.parquet` (main file)
One row per row of `data/test_data.parquet`, **same order** (656,601 rows). Join on `row_id` (= position in `test_data.parquet`, starting at 0); `vhcl_anon_id` + `vhcl_fct_taken_datm` are not unique because there are several rows on the same day.

| Column | Content | Filled for |
|---|---|---|
| `row_id` | position in `test_data.parquet` | all rows |
| `vhcl_anon_id`, `vhcl_fct_taken_datm` | key (original) | all rows |
| `vhcl_current_mileage` | reported mileage (original) | 493,554 rows (until 1 Jul 2026) |
| `forecast_mileage` | **predicted mileage** (forecast) | the 163,047 empty rows (2 Jul – 11 Nov 2026) |
| `forecast_method` | `model` (gradient boosting, 159,302 rows) or `cold_start` (age curve for 149 new cars without history, 3,745 rows) | empty rows |
| `mileage_error_flag` | **0/1: reading is conspicuous** | rows with a reported mileage |
| `flag_score` | strength of the anomaly (0 = inconspicuous; rule violations are doubled) | rows with a reported mileage |
| `flag_reason` | plain-text reason | flagged rows (5,338) |

Forecasts are non-decreasing over time per vehicle and are made from the readings known until 1 Jul 2026.

There is **no per-row nowcast** in this file: the empty test rows have no rental status, and no row-wise nowcast was built for the reported rows. The nowcast model (`models/nowcast_status_2026-07-01.joblib`) and its results on held-out vehicles (`results/nowcast_metrics.csv`) are part of the submission.

## Convenience files
- `predictions_forecast.csv`: only the empty rows (`vhcl_anon_id`, `vhcl_fct_taken_datm`, `predicted_mileage`, `method`).
- `../data/processed/partb_flagged_rows.csv`: only the flagged rows with score and reason.
- `slides.pdf`: five slides with the results.

## Reproduce
```bash
./submission/run.sh submission    # from Project_2/: error flags, forecast, merged file, slides
```
Details and results: `SUBMISSION.md`.
