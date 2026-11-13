# Submission: Project 2 – Mileage Forecasting & Erroneous Mileage Detection

## Authors
**Team name:** <enter team name> (single-person team)

**Project:** Project 2: Mileage Forecasting & Erroneous Mileage Detection

## Summary: what we deliver

**One table, `submission/predictions.parquet`**, with one row per row of `test_data.parquet` (656,601 rows, same order, key `row_id`):

- `forecast_mileage` for all **163,047** empty rows (2 Jul – 11 Nov 2026): 159,302 by the model, 3,745 by a cold-start rule for 149 new cars without any history.
- `mileage_error_flag` (0/1) with `flag_score` and `flag_reason` for all **493,554** reported readings: **5,338 flagged (1.08 %)**.

Column reference: [`submission/README.md`](submission/README.md). Five slides: [`submission/slides.pdf`](submission/slides.pdf).

| Deliverable | Result |
|---|---|
| **Forecast**, 133 days ahead, backtest on the real test fleet (4,224 vehicles, forecast from 18 Feb 2026, compared with the readings known until 1 Jul 2026) | **MAE 1,396 km** vs 1,549 km best rule baseline (9.9 % better) and 5,053 km naive last-value baseline (72 % better) |
| Forecast, held-out vehicles, 3 cut-off dates | MAE 1,340 – 1,536 km; 5.0 – 11.8 % better than the best rule baseline |
| **Nowcast**, up to 28 days, with rental status | MAE 590 – 666 km vs 677 – 757 km best rule baseline (10 – 13 % better) |
| New cars without history (149 in the test file) | cold-start age curve, MAE 961 – 1,255 km vs 1,091 – 1,312 km for the fleet-rate rule |
| **Error flags** | 5,338 of 493,554 reports flagged; recall 99.3 % on 6,000 injected errors (97.0 – 100 % per error type) |

No ground truth exists for the hidden test period (2 Jul – 11 Nov 2026), so all accuracy numbers come from vehicles and periods held out from training and tuning, not from the hidden period itself.

## Problem and approach
The test file contains 163,047 rows without a mileage (all between 2 Jul and 11 Nov 2026, no rental status), history until 1 Jul 2026, and vehicles that do not appear in the sample. This is a **133-day forecast on unseen vehicles**; the nowcast and the error flags are covered as separate parts.

- **Forecast** (`src/forecast.py`, `src/tuning.py`, `src/predict.py`): gradient boosting on the mileage increment since the last known reading (target `km − last km`, at least 0, absolute-error loss). Features: gap in days, vehicle rates over 30/90/180 days and the full history, expected increments of the baselines, vehicle attributes, month. Predictions are made monotone per vehicle over time (isotonic regression).
- **Cold start** (`src/coldstart.py`): for vehicles without any reading, start at a typical new-car reading (about 12 km) and follow the median growth curve of new vehicles by age in days.
- **Nowcast** (`src/nowcast.py`): mileage on a day without report (up to 28 days after the last reading) from the last reading and the rental status of that day (rental running, product, one-way, days since handover, planned return, repair, planned utilisation).
- **Error flags** (`src/anomaly.py`): five flags combined with OR: *rollback* (reading below the previous one; the longest non-decreasing subsequence marks the outlier itself), *too high* (above all later readings), *speed* (more than 1,500 km/day), *high* / *low* (increase far above or below the expected usage for that time gap; thresholds are quantiles per time gap, fitted on the sample). Every flag has a score and a plain-text reason.

**Assumptions:** per vehicle and day the highest reported mileage counts; readings more than 10,000 km away from the local median are ignored for modelling; the planned return date is treated as known at the time of the row (it is never before the row date in the data).

## Models and why
All models run locally on CPU and are open source (scikit-learn, pandas). No external model and no API key.

| Model | Open / closed | Runs where | Main metric (MAE km, held-out vehicles) | Time | Chosen? |
|---|---|---|---|---|---|
| Rule: last value stays (`b_last`) | open | local | 4,659 – 5,418 | seconds | no, lower bound |
| Rule: class median rate × days (`b_segment`) | open | local | 1,489 – 1,618 | seconds | baseline |
| Gradient boosting, untuned | open | local | 1,350 – 1,547 | about 1 min | no |
| **Gradient boosting, tuned** | open | local | **1,340 – 1,536** | about 2 min | **yes** |
| Tuned model + test-fleet history in training | open | local | 1,400 (backtest) vs 1,396 without | about 2 min | no, no gain |
| Chain nowcast → forecast | open | local | equal or worse than forecast alone | about 3 min per cut-off | no |
| New-car age curve | open | local | 961 – 1,255 (rule: 1,091 – 1,312) | seconds | yes, new cars only |

(Cut-off order of the ranges: 1 Feb 2026 / 1 Oct 2025 / 1 Jul 2025.)

**Comparison rules:** *naive* = last known mileage. *Best rule* = last known mileage + class rate × days since the last reading, where the class rate is the median km/day of the vehicles of the same class (Standard, Intermediate, Full Size) over the 90 days before the cut-off date.

## Results
All numbers on **held-out vehicles** (15 % of the sample vehicles, never used for training or tuning), 133 days after the cut-off date, training only on targets that end before the cut-off. Evaluation tables: `results/`. Feature importance: `reports/figures/forecast_feature_importance.png`.

**Forecast, held-out vehicles (MAE in km)**

| Cut-off | Naive (last value) | Best rule | Model | Model vs best rule |
|---|---|---|---|---|
| 1 Jul 2025 (same season as the test period) | 5,418 | 1,618 | **1,536** | 5.0 % |
| 1 Oct 2025 | 4,751 | 1,521 | **1,342** | 11.8 % |
| 1 Feb 2026 | 4,659 | 1,489 | **1,340** | 10.0 % |

**Forecast, backtest on the real test fleet** (4,224 test vehicles, 97,514 readings; forecast from 18 Feb 2026 = 133 days before the last known date, compared with the readings known until 1 Jul 2026; training on sample vehicles only):

| Horizon | Naive | Best rule | Model |
|---|---|---|---|
| 0 – 14 days | 863 | 484 | **432** |
| 31 – 60 days | 3,558 | 1,250 | **1,126** |
| 91 – 133 days | 8,236 | 2,260 | **2,046** |
| all | 5,053 | 1,549 | **1,396** (WAPE 6.8 %, bias −186 km) |

**Nowcast (MAE in km, up to 28 days):**

| Cut-off | Best rule | Model, history only | Model + rental status |
|---|---|---|---|
| 1 Jul 2025 | 757 | 711 | **666** |
| 1 Oct 2025 | 707 | 668 | **636** |
| 1 Feb 2026 | 677 | 609 | **590** |

**Error flags:** 5,338 flagged reports: 2,781 by the rules only, 2,257 by the deviation check only, 300 by both. Injected-error test (6,000 readings changed): digit appended ×10 100 %, +5,000 to +30,000 km 99.9 %, reset to 0 99.8 %, −10,000 to −30,000 km 99.7 %, digit dropped ÷10 99.4 %, −20 % to −60 % 97.0 %; overall 99.3 %. Flagged rows with reasons: `data/processed/partb_flagged_rows.csv`.

**What did not work / limits (honest):**
- Tuning gains only 1.1 % over the untuned configuration (1,365 vs 1,380 km on validation). The model is essentially "class rate × time gap": the feature importance is dominated by `exp_segment` (518 km) and `gap` (501 km), everything else is below 50 km.
- In the cut-off of the same season as the test period (1 Jul 2025) the model underestimates (bias −516 km) and its lead over the best rule is smallest (5.0 %). A seasonal feature was not tested.
- The gain over the best rule varies with the holdout and the date (5 % to 12 %).
- Adding the test fleet's own history to the training did not help (MAE 1,400 vs 1,396 km), so it is not used. Chaining the nowcast into the forecast gave no gain either.
- The cleaning (outlier removal, monotone series) runs once over the whole time series, a small leak in the cut-off experiments; targets in the experiments are the cleaned readings, the hidden truth is probably raw, so the numbers are probably slightly optimistic.
- Error flags: injected errors are large; subtle manipulation (a few percent) is not covered. The false-alarm rate on untouched readings is 1.85 %, which includes real natural oddities; without labels we cannot say how many are wrong reports. The expected value is a simple rate (vehicle or class median), not the trained model.
- No nowcast is delivered per test row: the empty test rows have no rental status.

## How to run
Python 3.11+ and about 12 GB RAM; run from `Project_2/`. The raw files `data_30_percent_sample.parquet` and `test_data.parquet` are in `Project_2/data/`. No environment variables, no secrets.

```bash
cd Project_2
./run.sh setup        # once: create .venv and install requirements
./run.sh submission   # rebuild all submission files from the saved parameters (about 2 min 10 s)
./run.sh all          # everything incl. tuning and all evaluations (about 40 min)
./run.sh partb predict export slides   # or single steps, in any order
```

| Step | What it does / output | Time |
|---|---|---|
| `setup` | virtual environment and packages | about 1 min |
| `explore` | data overview → `data_overview.txt` (not tracked) | 21 s (measured) |
| `baselines` | rule baselines → `results/baseline_metrics.csv` | 8 s (measured) |
| `coldstart` | new-car rule vs fleet rate → `results/coldstart_metrics.csv` | 13 s (measured) |
| `nowcast` | nowcast with and without status → `results/nowcast_metrics.csv` | about 6 min (measured) |
| `tuning` | random search, bias, holdout, feature importance → `results/`, `reports/figures/` | about 25 min (measured) |
| `save_nowcast` | train and save the nowcast model → `models/nowcast_status_2026-07-01.joblib` | 2 min 25 s (measured) |
| `backtest` | backtest on the test fleet → `results/backtest_test_fleet.csv` | about 5 min (estimate) |
| `partb` | thresholds, injected-error test, flags → `results/partb_*.csv`, `data/processed/partb_flagged_rows.csv` | 19 s (measured) |
| `predict` | train the final forecast model, predict the empty test rows → `submission/predictions_forecast.csv`, `models/forecast_final_2026-07-01.joblib` | 1 min 49 s (measured) |
| `export` | merge forecast and flags → `submission/predictions.parquet` | 2 s (measured) |
| `slides` | build `submission/slides.pdf` from `results/` | 1 s (measured) |

Times were measured on 4 CPU cores and 16 GB RAM, one step at a time; "estimate" marks a step that was not timed on its own. `submission` runs `partb`, `predict`, `export`, `slides` in that order. The old per-step commands still work, e.g. `python -m scripts.run_partb`.

## Demo
`submission/slides.pdf` (5 slides, English). Open it with any PDF viewer, for example `xdg-open submission/slides.pdf` on Linux, or directly on GitHub. `./run.sh slides` rebuilds it from the result tables.

## Next steps
- Seasonal index, fleet trend, remaining kilometres to the contract limit and branch features for the forecast (the feature importance shows that almost only "rate × gap" counts today).
- Cross-validation over vehicles instead of a single split.
- Error flags with the nowcast as expected value (out-of-fold, with quantile uncertainty bands) and tests with subtle manipulation.
- For SIXT: run the flag list regularly to review conspicuous readings before valuation and defleet decisions.
