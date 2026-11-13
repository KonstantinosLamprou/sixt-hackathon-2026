# Teil B: Erkennung fehlerhafter Kilometermeldungen (einfache Form)

Code: `src/anomaly.py`, Start: `python -m scripts.run_partb`. Ergebnisse: `results/partb_summary.csv`, `results/partb_injection.csv`, markierte Zeilen: `data/processed/partb_flagged_rows.csv`.

## Ansatz
Eine Meldung ist der höchste Stand eines Fahrzeugs an einem Tag. Fünf Flags, die zu `flag = 1` zusammengefasst werden (ODER):

| Flag | Bedeutung |
|---|---|
| `f_rollback` | Stand liegt unter dem Stand davor, obwohl spätere Messungen darüber liegen. Bestimmt über die **längste nicht fallende Teilfolge**: markiert den Ausreißer selbst, nicht die Kaskade danach. |
| `f_toohigh` | Stand liegt über allen späteren Messungen |
| `f_speed` | mehr als 1.500 km/Tag zwischen zwei plausiblen Messungen |
| `f_high` | Zuwachs deutlich über dem Erwartbaren (Median-Rate des Fahrzeugs × Tage; bei weniger als 5 Intervallen die Median-Rate der Fahrzeugklasse) |
| `f_low` | Zuwachs deutlich darunter (mögliche Untermeldung) |

Die Schwellen für `f_high`/`f_low` sind das 99,5 %- bzw. 0,2 %-Quantil der Abweichungen **je Zeitabstand** (1, 2, 3–4, 5–7, 8–14, 15–28, über 28 Tage), bestimmt auf dem Sample und auf die Testdaten angewendet. Zu jedem Flag gibt es Score und Begründungstext.

Gegenüber dem ursprünglichen Vorschlag wurden bewusst geändert: Rückwärts-Tacho über die längste nicht fallende Teilfolge (das einfache `decrease` flaggt in 78 % der Fälle Folgen von Messungen nach einem einzelnen Ausreißer), Abweichungen horizont-abhängig statt fester 2.000 km oder oberer 5 %, beide Richtungen. Die Regel "Zuwachs bei nicht vermietetem Fahrzeug" wurde nicht übernommen: Nur 88 von 164.113 Zuwächsen über 1.000 km enden bei einem Fahrzeug, das an beiden Messpunkten nicht vermietet war.

## Ergebnis
- Sample: 1,85 % der Meldungen geflaggt, Testdatei: 1,53 % der Meldungen, **5.338 von 493.554 Zeilen (1,08 %)** mit gemeldetem Stand.
- **Prüfung mit eingespritzten Fehlern** (6.000 Meldungen im Sample verfälscht: Verzehnfachung, Zehntel, Null, +5.000 bis 30.000 km, −20 bis −60 %, −10.000 bis −30.000 km): **Recall 99,3 %** (Spanne je Art 97,0 bis 100 %).

## Einordnung gegenüber der Aufgabenstellung
Die README nennt für Teil B **zwei brauchbare Ansätze, die sich ergänzen sollen** (Abweichung vom Prädiktor; physikalisch unmögliche Fälle als schwache Labels zur Plausibilisierung). Beide sind enthalten, aber unterschiedlich weit umgesetzt:

- **Regelbasiert:** voll umgesetzt (`f_rollback`, `f_toohigh`, `f_speed`).
- **Abweichungsbasiert:** nur teilweise. Die README sieht vor, den **trainierten Prädiktor** zu nutzen und Meldungen zu flaggen, die davon abweichen. Wir vergleichen stattdessen mit einer **einfachen Rate** (Median des Fahrzeugs bzw. der Fahrzeugklasse mal Tage). Das ist eine abgespeckte Variante, entstanden aus Zeitgründen.

**Überschneidung der beiden Ansätze** (5.338 geflaggte Testzeilen):

| | Zeilen |
|---|---|
| nur durch Regeln geflaggt | 2.781 |
| nur durch Abweichung geflaggt | 2.257 |
| von beiden geflaggt | 300 |

Nur etwa 10 % der Regel-Flags löst auch die Abweichungsprüfung aus. Das kann bedeuten, dass die Ansätze **verschiedene Fehlerarten** finden (ergänzen sich), oder dass die Abweichungsprüfung viel Rauschen enthält. Ohne Labels lässt sich das nicht unterscheiden; hier wäre der trainierte Erwartungswert (Nowcast) der nächste Schritt.

## Einschränkungen (wichtig)
- Die eingespritzten Fehler sind **groß und auffällig**. Subtile Manipulationen (wenige Prozent, wenige hundert km) sind damit nicht geprüft; dort ist die Erkennung deutlich schlechter zu erwarten.
- Die Fehlalarmrate von 1,85 % auf nicht verfälschten Meldungen enthält auch **echte natürliche Auffälligkeiten**; wir haben keine Labels und wissen nicht, wie viele davon wirklich falsch gemeldet sind.
- Der Erwartungswert kommt aus einer einfachen Rate (Fahrzeug- bzw. Klassenmedian), **nicht** aus dem trainierten Nowcast-Modell. Ein Nowcast-Erwartungswert mit Status wäre schärfer (nächster Schritt).
- Mehrere Originalzeilen mit identischem Tagesstand erscheinen alle als geflaggt (daher einzelne Doppelzeilen in der Liste).
- Das Abgabeformat der Jury (z. B. Spaltenname) ist ungeklärt.
