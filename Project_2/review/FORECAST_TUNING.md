# Forecast: Tuning, Holdout-Ergebnis und Feature-Wichtigkeit

Code: `src/tuning.py`, Start: `python -m scripts.run_tuning`. Rohdaten der Läufe: `results/tuning_forecast.csv`, `results/forecast_tuned_holdout.csv`, `results/forecast_feature_importance.csv`, `results/best_forecast_params.json`. Modell: `models/forecast_tuned_2026-02-01.joblib`. Diagramm: `reports/figures/forecast_feature_importance.png`.

## Aufbau
- **Fahrzeug-Split 70 / 15 / 15** (19.511 Training, 4.144 Validierung, 3.967 Holdout). Der Holdout wurde erst nach der Auswahl benutzt.
- **Zufallssuche** über 12 Konfigurationen (ungetunt plus 11 zufällige), Bewertung = mittlerer MAE auf den Validierungs-Fahrzeugen an den Stichtagen 01.10.2025 und 01.02.2026. Der saisonale Stichtag 01.07.2025 wurde nicht zum Tuning benutzt.
- **Bias-Faktor** auf den vorhergesagten Zuwachs und **Status-Historie-Features** (Anteil Miet- und Langzeittage der letzten 30/90 Tage, mittlere Mietdauer, letzter `uti_plan`): Übernahme nur bei mindestens 1 % besserem Validierungs-MAE.

## Ergebnis der Suche
| | Validierungs-MAE (km) |
|---|---|
| ungetunte Konfiguration | 1.380 |
| beste Konfiguration (lr 0,03; 300 Iterationen; 128 Blätter; Mindestblatt 1.000; L2 = 10; max_features 0,8) | **1.365** (1,1 % besser) |
| Spanne der übrigen Konfigurationen | 1.366 bis 1.380 |
| Bias-Faktor | beste Wert 1,0, also keine Skalierung |
| mit Status-Historie-Features | 1.355 (0,7 % besser, unter der Schwelle von 1 %, **verworfen**) |

Das Tuning bringt wenig. Die Einstellungen sind nicht der Engpass.

## Holdout (MAE in km, 133 Tage Horizont)
| Stichtag | Stand bleibt | beste Regel `b_segment` | Modell ungetunt | **Modell getunt** | getunt gegen `b_segment` |
|---|---|---|---|---|---|
| 01.07.2025 (gleiche Jahreszeit wie der echte Test) | 5.418 | 1.618 | 1.547 | **1.536** | 5,0 % besser |
| 01.10.2025 | 4.751 | 1.521 | 1.362 | **1.342** | 11,8 % besser |
| 01.02.2026 | 4.659 | 1.489 | 1.350 | **1.340** | 10,0 % besser |

- Der Vorsprung des getunten gegenüber dem ungetunten Modell ist klein (0,7 bis 1,5 %).
- **Beim saisonalen Stichtag schätzt das Modell deutlich zu niedrig** (Bias −516 km, die Regel `b_segment` nur −84 km). Der Bias-Faktor wurde auf den Stichtagen Oktober und Februar bestimmt, wo die Verzerrung kleiner ist. Das spricht dafür, dass dem Modell die Jahreszeit fehlt.

## Feature-Wichtigkeit (Permutation, Validierung, Anstieg des MAE in km)
`exp_segment` 518, `gap` 501, `exp_rate_all` 45, `stale_days` 39, `exp_rate_90` 22, `target_month` 10, `exp_rate_30` 7, `last_km` 7, `cubic_capacity_liter` 6, `rate_all` 6.

Das Modell beruht im Wesentlichen auf "Segment-Rate mal Abstand" und korrigiert nur wenig. Fahrzeugmerkmale tragen kaum etwas bei. Weitere Verbesserung ist deshalb von **neuen Informationsquellen** zu erwarten (Saison, Flottentrend, Filiale, Restkilometer), nicht von mehr Tuning.

## Einschränkungen
- Die Zahlen gelten für einen anderen Holdout als die früheren Tabellen (15 % statt 20 % der Fahrzeuge, anderer Zufallsschnitt). Beim Stichtag 01.07.2025 beträgt der Vorsprung auf `b_segment` hier 5,0 % statt der früheren 2,7 %. Die Streuung zwischen Holdouts ist also nicht klein.
- Ein Holdout, ein Zufallssplit. Eine Kreuzvalidierung wäre belastbarer.
- Frühere Version der Parameterdatei enthielt fälschlich den MAE der verworfenen Zusatz-Features unter `val_mae_tuned`; behoben.
