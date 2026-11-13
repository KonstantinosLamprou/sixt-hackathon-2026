# Kette Nowcast -> Forecast: Test und Ergebnis

## Idee
Die letzte Messung ist oft veraltet (bei 28 % der Testfahrzeuge über 7 Tage, bei 16 % über 14 Tage). Wir schätzen zuerst den Stand am Stichtag (Nowcast), setzen ihn als Startpunkt und lassen den Forecast von dort laufen. Code: `chain.py`.

Der Status am Stichtag ist nicht direkt bekannt (nicht an jedem Tag gibt es eine Zeile). Wir schreiben den Status der letzten Zeile fort: Miete läuft weiter, solange das geplante Rückgabedatum nicht überschritten ist.

## Ergebnis (MAE in km, Holdout-Fahrzeuge)

| Stichtag | Teilmenge | beste Regel | Forecast allein | Kette ohne Status | Kette mit Status |
|---|---|---|---|---|---|
| 01.07.2025 | alle | 1.563 | 1.521 | **1.518** | 1.525 |
| 01.07.2025 | letzte Messung >14 Tage alt | 2.416 | 2.210 | 2.210 | **2.195** |
| 01.10.2025 | alle | 1.480 | 1.348 | **1.342** | 1.349 |
| 01.10.2025 | letzte Messung >14 Tage alt | 2.294 | 1.996 | **1.959** | 1.974 |
| 01.02.2026 | alle | 1.497 | **1.369** | 1.389 | 1.388 |
| 01.02.2026 | letzte Messung >7 Tage alt | 1.960 | **1.670** | 1.715 | 1.730 |

## Entscheidung
Die Kette bringt keinen verlässlichen Gewinn. Zwei Stichtage sind minimal besser (unter 1 %), der dritte ist schlechter (bis 3,6 % bei veralteten Messungen). Wir bauen sie **nicht** in die Abgabe ein. Der Forecast deckt die Lücke bis zum Stichtag schon über die Features `gap` und `stale_days` ab.

Der Nowcast bleibt als eigener Teil von Aufgabe A bestehen (siehe `NOWCAST.md`) und ist die Grundlage für Teil B.

## Wichtiger Nebenbefund
Im Stichtag 01.07.2025 (gleiche Jahreszeit wie der echte Test) schlägt der Forecast die beste Regel nur um 2,7 % (1.521 gegen 1.563 km), bei den anderen Stichtagen um 8 bis 9 %. Der Vorsprung auf der echten Testdatei kann kleiner sein als bisher angenommen.
