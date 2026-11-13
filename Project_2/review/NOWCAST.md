# Nowcast

## Was der Nowcast ist
Schätzt den Kilometerstand eines Fahrzeugs **an einem Tag, für den die Meldung noch fehlt** (bis zu 28 Tage Verzögerung), aus dem letzten bekannten Stand **und dem Status des Zieltages**: läuft eine Miete, welches Produkt, Einweg, seit wann, bis wann geplant, Reparatur, `uti_plan`.

## Leakage-Prüfung über `rntl_return_date`
Im Sample liegt das Return-Datum in **allen** OnRent-Zeilen am Messtag oder danach (nie davor), in 31 % in der Zukunft. Es ist daher das zum Zeitpunkt der Zeile bekannte, geplante Datum und als Feature zulässig. (Das Handover-Datum liegt nie nach dem Messtag.)

## Aufbau (`nowcast.py`)
- Gleiche Pipeline wie der Forecast, aber Horizont 28 Tage, Trainings-Stichtage alle 14 Tage, 4 Zeilen je Fahrzeug und Stichtag.
- Zwei Modelle auf denselben Zeilen: **ohne** Status (nur Historie) und **mit** Status (`days_since_handover`, `days_to_return`, `rental_len`, `on_rent`, `longterm`, `one_way`, `repair`, `status_code`, `uti_plan`, `n_rows_day`). Der Unterschied zeigt, was der Status bringt.
- Bewertung nur auf Holdout-Fahrzeugen, Stichtage 01.07.2025, 01.10.2025, 01.02.2026.

## Ergebnis (MAE in km, alle Horizonte bis 28 Tage)

| Stichtag | `b_last` | `b_segment` (beste Regel) | Modell ohne Status | **Modell mit Status** |
|---|---|---|---|---|
| 01.07.2025 | 1.534 | 757 | 711 | **666** |
| 01.10.2025 | 1.529 | 707 | 668 | **636** |
| 01.02.2026 | 1.337 | 677 | 609 | **590** |

Der Status senkt den Fehler gegenüber dem reinen Historien-Modell um 3 bis 6 %, gegenüber der besten Regel um 10 bis 13 %.

## Einschränkungen
- Der Gewinn durch Status ist moderat. Wir haben keine Feature-Wichtigkeit gemessen.
- Hyperparameter sind nicht getunt.
- Pseudo-Test: Zielstand ist die bereinigte Reihe, der Bias des Modells ist leicht negativ (−100 bis −240 km).
- Die Testdatei prüft den Nowcast vermutlich nicht (leere Zeilen haben keinen Status). Ob und wie er bewertet wird, ist bei den Organisatoren zu erfragen.
