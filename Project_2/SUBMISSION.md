# Submission: Project 2 – Mileage Forecasting & Erroneous Mileage Detection

## Authors
**Team name:** <Teamname eintragen>

**Project:** Project 2: Mileage Forecasting & Erroneous Mileage Detection (Einzelteam)

## Problem und Ansatz
Für Langzeitmieten liegt der Kilometerstand oft erst verzögert vor. Wir sagen den Stand für die leeren Zeilen von `test_data.parquet` voraus (Teil A) und markieren auffällige gemeldete Stände (Teil B).

**Was die Testdatei verlangt (aus den Daten belegt):** 163.047 Zeilen (24,8 %) ohne Kilometerstand, alle zwischen 02.07. und 11.11.2026 (133 Tage), ohne Mietstatus. Die Historie reicht bis 01.07.2026. Die Testfahrzeuge kommen im Sample nicht vor. Die Aufgabe ist also ein **Forecast über 133 Tage** auf neuen Fahrzeugen.

**Teil A, Forecast (`src/forecast.py`, `src/tuning.py`, `src/predict.py`):** Gradient Boosting auf dem Kilometer-Zuwachs seit der letzten bekannten Messung (Ziel: `km − letzter Stand`, mindestens 0, Verlust: absoluter Fehler). Features: Abstand in Tagen, Fahrzeugraten (30/90/180 Tage, gesamte Historie), erwarteter Zuwachs der Baselines, Fahrzeugmerkmale, Monat. Vorhersagen werden pro Fahrzeug über die Zeit monoton gemacht (isotone Regression).
**Neufahrzeuge ohne Historie (149 Fahrzeuge, 2,3 % der Zeilen):** Alterskurve (Startstand ca. 12 km plus mediane Wachstumskurve neuer Fahrzeuge) (`src/coldstart.py`).
**Teil A, Nowcast (`src/nowcast.py`):** Stand an einem Tag ohne Meldung (bis 28 Tage) aus letztem Stand und Mietstatus des Tages. Die Kette Nowcast → Forecast brachte keinen Gewinn und ist nicht Teil der Abgabe.
**Teil B (`src/anomaly.py`):** Fünf Flags: Rückwärts-Tacho (längste nicht fallende Teilfolge), zu hoch gegenüber späteren Messungen, mehr als 1.500 km/Tag, Zuwachs deutlich über oder unter dem Erwartbaren je Zeitabstand. Schwellen aus dem Sample.

**Annahmen:** Pro Fahrzeug und Tag zählt der höchste gemeldete Stand. Ausreißer über 10.000 km Abweichung vom lokalen Median werden für die Modellierung ignoriert. Das Abgabeformat ist nicht bestätigt; wir liefern `submission/predictions_forecast.csv` (Fahrzeug, Datum, Vorhersage, Methode) und `data/processed/partb_flagged_rows.csv`.

Entscheidungen und Begründungen: `review/ENTSCHEIDUNGEN.md`, `review/FORECAST_TUNING.md`, `review/NOWCAST.md`, `review/CHAIN.md`, `review/COLDSTART.md`, `review/TEIL_B.md`.

## Modelle und warum
Alles läuft lokal auf CPU, alle Verfahren sind Open Source (scikit-learn, pandas). Kein externes Modell, kein API-Schlüssel.

| Modell | Open/Closed | Läuft wo | Hauptmetrik (Holdout, MAE km, 133 Tage) | Zeit | Gewählt? |
|---|---|---|---|---|---|
| Regel: Stand bleibt (`b_last`) | open | lokal | 4.659 bis 5.418 | Sekunden | nein, untere Latte |
| Regel: Median-Rate der Fahrzeugklasse (`b_segment`) | open | lokal | 1.489 / 1.521 / 1.618 | Sekunden | Baseline |
| Gradient Boosting, ungetunt | open | lokal | 1.350 / 1.362 / 1.547 | ca. 1 min | nein |
| **Gradient Boosting, getunt** | open | lokal | **1.340 / 1.342 / 1.536** | ca. 2 min | **ja** |
| Kette Nowcast → Forecast | open | lokal | gleich oder schlechter | ca. 3 min je Stichtag | nein |
| Neufahrzeug-Alterskurve | open | lokal | 961 bis 1.255 (Regel: 1.091 bis 1.312) | Sekunden | ja, nur Neufahrzeuge |

(Reihenfolge der Zahlen: Stichtag 01.02.2026 / 01.10.2025 / 01.07.2025.)

## Ergebnisse
Alle Zahlen auf **zurückgehaltenen Fahrzeugen** (15 % der Sample-Fahrzeuge, nie im Training, nie zum Tuning benutzt), 133 Tage nach dem Stichtag, Training nur mit Daten bis 133 Tage vor dem Stichtag.

| Stichtag | Stand bleibt | beste Regel | Modell getunt | Vorsprung auf beste Regel |
|---|---|---|---|---|
| 01.07.2025 (gleiche Jahreszeit wie der echte Test) | 5.418 | 1.618 | **1.536** | 5,1 % |
| 01.10.2025 | 4.751 | 1.521 | **1.342** | 11,8 % |
| 01.02.2026 | 4.659 | 1.489 | **1.340** | 10,0 % |

- **Nowcast** (Horizont bis 28 Tage, mit Status): MAE 590 bis 666 km gegen 677 bis 757 km der besten Regel (10 bis 13 % besser).
- **Teil B:** Auf der Testdatei 5.338 von 493.554 Zeilen mit gemeldetem Stand geflaggt (1,08 %). Bei 6.000 eingespritzten Fehlern (Verzehnfachung, Zehntel, Null, ±Beträge) Recall 99,3 %.

**Was nicht funktioniert hat / Grenzen (ehrlich):**
- Das Tuning bringt nur 1,1 % gegenüber der ungetunten Konfiguration. Das Modell besteht im Kern aus „Klassen-Rate × Abstand".
- Im saisonal passenden Stichtag (01.07.2025) schätzt das Modell deutlich zu niedrig (Bias −516 km). Ein Saison-Index wurde nicht mehr getestet.
- Der Holdout ist ein Zufallsschnitt; die Streuung zwischen Holdouts ist spürbar (Vorsprung beim Juli-Stichtag war auf einem anderen Holdout 2,7 %).
- Die Bereinigung wurde einmal über die gesamte Zeitreihe gerechnet, das ist eine kleine Lücke im Pseudo-Test. Die Zielstände sind bereinigt, die echte Wahrheit vermutlich roh. Die Zahlen sind daher eher optimistisch.
- Das Endmodell für die Testdatei wurde auf Sample **und** bekannter Testhistorie trainiert und nicht separat validiert. Die isotone Glättung wurde nicht auf dem Holdout gemessen.
- Teil B: Die eingespritzten Fehler sind groß und auffällig. Für subtile Manipulationen ist die Erkennung nicht belegt. Die Fehlalarmrate von 1,85 % auf nicht verfälschten Meldungen enthält auch echte natürliche Auffälligkeiten; ohne Labels wissen wir nicht, wie viele falsch gemeldet sind. Der Erwartungswert kommt aus einer einfachen Rate, nicht aus dem Nowcast-Modell.

## So läuft es
Python 3.11+, aus einer frischen Kopie. Die Rohdaten (`data_30_percent_sample.parquet`, `test_data.parquet`) liegen in `Project_2/data/`. Keine Umgebungsvariablen nötig.

```bash
cd Project_2
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m scripts.run_baselines       # Baselines, ca. 10 s
python -m scripts.run_tuning          # Forecast tunen + Holdout + Feature-Wichtigkeit, ca. 25 min
python -m scripts.run_nowcast         # Nowcast, ca. 6 min
python -m scripts.run_coldstart       # Neufahrzeug-Regel, ca. 1 min
python -m scripts.run_partb           # Teil B inkl. Test mit eingespritzten Fehlern, ca. 20 s
python -m scripts.run_predict_test    # Vorhersage fuer die Testdatei, ca. 2 min -> submission/predictions_forecast.csv
```
Benötigt etwa 8 bis 12 GB RAM für die Läufe auf dem Sample.

## Demo
`submission/slides.pdf` (max. 5 Folien) oder `submission/demo.mp4` (1 min): <noch zu erstellen>

## Nächste Schritte
- Saison-Index, Flottentrend, Restkilometer bis zur Vertragsgrenze und Filialmerkmale als Features testen (die Feature-Wichtigkeit zeigt, dass bisher fast nur „Rate × Abstand" zählt).
- Eine Kreuzvalidierung über Fahrzeuge statt eines einzelnen Splits.
- Teil B mit dem Nowcast-Modell als Erwartungswert und Unsicherheitsbändern (Quantil-Modelle) statt einfacher Rate, und eine Prüfung mit subtilen Fehlern.
- Für SIXT: Wöchentlicher Lauf der Flag-Liste zur Prüfung auffälliger Meldungen vor Bewertungs- und Defleet-Entscheidungen. Offen: Abgabeformat und Baseline der Jury klären.
