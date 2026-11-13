# Neustart-Fall (Cold Start)

## Problem
In `test_data.parquet` haben 149 Fahrzeuge (3.745 Zeilen, 2,3 % der zu füllenden Zeilen) **gar keinen bekannten Kilometerstand**. Es sind Neufahrzeuge (Baujahr 2026/27), deren erste Zeile zwischen 02.07. und 19.09.2026 liegt. Die bisherige Pipeline (Inner Join mit der Historie) ließ sie wegfallen. Es braucht aber für **jede** leere Zeile eine Vorhersage.

## Entscheidung
`Vorhersage = typischer Startstand + mediane Wachstumskurve neuer Fahrzeuge nach Alter` (Alter = Tage seit der ersten Zeile des Fahrzeugs). Code: `coldstart.py`.

- **Startstand (ca. 12 km):** Im Sample liegt der erste Stand neuer Fahrzeuge im Median bei 12 km, bei 90 % unter 500 km. `last_km = 0` wäre also fast richtig; wir nehmen den gemessenen Median.
- **Wachstumskurve:** Median des Zuwachses seit der ersten Messung, für jedes Alter 0 bis 133 Tage, aus allen Neufahrzeugen im Sample. Beispiel: nach 30 Tagen ca. 2.300 km, nach 90 Tagen ca. 7.100 km, nach 133 Tagen ca. 10.200 km.

## Warum eine Kurve und nicht "Start 0 + Segment-Rate"
Getestet im Pseudo-Test mit simulierten Neufahrzeugen (nur Holdout, drei Stichtage), `evaluate_coldstart.py`:

| Stichtag | Stand bleibt 0 | Start 0 + Segment-Rate × Tage | **Alterskurve** |
|---|---|---|---|
| 01.07.2025 | MAE 4.284 | 1.312 | **1.255** |
| 01.10.2025 | MAE 3.619 | 1.091 | 1.091 |
| 01.02.2026 | MAE 3.276 | 1.137 | **961** |

Die Alterskurve ist an einem Stichtag klar besser, an den anderen gleich gut. Sie ist außerdem einfach zu erklären.

## Einschränkungen
- Die Kurve ist ein Median ohne Saison und ohne Fahrzeugklasse. Eine Version pro Fahrzeugklasse war nicht besser.
- `vhcl_max_mileage` fehlt bei diesen Fahrzeugen komplett, deshalb wird kein Modell mit diesem Feature für sie benutzt.
- Der WAPE liegt bei ca. 30 %, weil die echten Werte klein sind. Der MAE (ca. 1.000 bis 1.250 km) ist mit dem restlichen Modell vergleichbar.
- Die Pseudo-Neufahrzeuge stammen aus dem Sample. Ob die 2026er Neufahrzeuge genauso schnell wachsen, wissen wir nicht.
- Die Aussage "SIXTs Pipeline bricht bei fehlenden Fahrzeugen ab oder bestraft sie" ist eine Vermutung und nicht belegt.
