# Project 2: Entscheidungen und Begründungen

Dieses Dokument erklärt, **was** bisher entschieden wurde, **warum**, welche **Alternativen** es gab und wo die **Schwachstellen** liegen. Es ist so geschrieben, dass du jede Entscheidung selbst nachvollziehen und gegenüber der Jury vertreten kannst.

Stand: Datenanalyse, Aufräumen, Messlatte (Baselines) und ein erstes trainiertes Modell. Nicht getan: Tuning, Vorhersage auf der echten Testdatei, Teil B (Fehlererkennung), Abgabe-Dokumente.

---

## 0. Die Aufgabe (aus `Project_2/README.md`)

**Problem:** Bei Langzeitmieten kennt SIXT den Kilometerstand oft nur bei Rückgabe, bei monatlicher Meldung bis zu 4 Wochen verspätet. Für Bewertung und Verkaufszeitpunkt (Defleet) braucht man ihn aber aktuell.

- **Teil A:** Aktuellen Stand schätzen (Nowcast) und zukünftigen Stand vorhersagen (Forecast).
- **Teil B:** Fehlerhafte Meldungen erkennen (absichtlich oder versehentlich). Es gibt keine Labels. Vorgeschlagen: Abweichung von der Vorhersage (Residuen) oder Regeln für physikalisch unmögliche Fälle (Stand fällt, zu hohe km/Tag).

**Bewertung laut README:** "error on held-out vehicles / periods (e.g. MAE, MAPE) for nowcast and forecast, compared to a naive baseline; and how plausible the flagged reports are."

Daraus folgt direkt, was wir brauchen:
1. Fehler auf **zurückgehaltenen Fahrzeugen und Zeiträumen** (nicht auf Trainingsdaten),
2. ein Vergleich mit einer **naiven Baseline**,
3. plausible Flags für Teil B.

Jede Entscheidung unten dient einem dieser drei Punkte.

---

## 1. Wie wir die Testdatei interpretiert haben

**Entscheidung:** `test_data.parquet` ist eine reine **Forecast-Aufgabe**.

**Beleg aus den Daten (gemessen, siehe `explore_data.py`):**
- In `test_data` fehlt bei 163.047 Zeilen (24,8 %) der Kilometerstand.
- Bei genau diesen Zeilen sind auch Mietstatus, Reparaturstatus und Produkt leer.
- Alle liegen zwischen 02.07.2026 und 11.11.2026 (133 Tage).
- Bei allen 8.099 betroffenen Fahrzeugen bilden sie einen zusammenhängenden Block am Ende der Zeitreihe.
- Die Test-Fahrzeuge kommen im Sample nicht vor (0 von 8.254).

**Folge:** Wir sagen aus der Historie bis 01.07.2026 die Stände bis 11.11.2026 voraus, **ohne** den zukünftigen Mietstatus zu kennen. Features, die nur in der Vergangenheit existieren (z. B. `uti_plan` der Zukunft, aktueller Mietstatus), dürfen wir für den Forecast nicht nutzen.

**Unsicherheit:** Ich habe nicht von den Veranstaltern bestätigt bekommen, in welchem Format sie Vorhersagen erwarten und ob die leeren Spalten `sixt_predicted_mileage_interp` und `sixt_error` dafür gedacht sind. Das musst du erfragen.

**Offen:** Den **Nowcast** (aktueller Stand anhand des aktuellen Mietstatus) haben wir noch nicht separat behandelt. Die Testdatei verlangt ihn nicht, die Aufgabenstellung nennt ihn aber. Entscheidung für später.

---

## 2. Datenaufbereitung (`data.py`)

### 2.1 Eine Messung pro Fahrzeug und Tag, mit dem Maximum

**Was:** Mehrere Zeilen am selben Tag werden zu einer, mit dem höchsten Stand.

**Warum:** In den Daten stehen am selben Tag oft zwei Zeilen (z. B. `OnRent` und `NotOnRent`) mit unterschiedlichen Ständen. Die `NotOnRent`-Zeile trägt typischerweise den Stand **nach** der Fahrt. Ohne Zusammenfassen sortiert man innerhalb eines Tages zufällig, und es entstehen scheinbare Rückwärtssprünge. Das hat die `starter.py`-Ausgabe belegt: Alle 12 Flags der Beispieldatei waren solche Artefakte (alle mit `days = 0`).

**Alternative:** Median oder letzter Wert des Tages. Das Maximum ist einfach und passt zur Beobachtung, dass der Stand nie fällt.

**Schwäche:** Steckt in einem Tag ein einzelner falscher, zu hoher Wert, gewinnt er.

### 2.2 Ausreißer (`spike`)

**Was:** Eine Messung gilt als `spike`, wenn sie mehr als **10.000 km** vom Median ihrer 7 Nachbarmessungen abweicht. Sie wird für die bereinigte Reihe ignoriert.

**Warum:** Die Daten enthalten offensichtlichen Unsinn (Maximum 999.999 km, Rückwärtssprünge bis −868.999 km, Raten bis 313.201 km/Tag). Solche Werte würden Raten und Training verfälschen.

**Woher die 10.000?** Das ist ein **erster geschätzter Wert**, kein abgestimmter. Ich habe ihn nicht systematisch gegen andere Werte getestet.

**Schwäche:** Kleinere Fehlmeldungen (z. B. +3.000 km) werden nicht erkannt.

### 2.3 Monotone Reihe (`decrease`) und `km`

**Was:** Der bereinigte Stand `km` ist der laufende Höchstwert. Fällt eine Messung mehr als 50 km darunter, wird sie als `decrease` markiert.

**Warum:** Ein Kilometerzähler fällt nicht. Alles, was fällt, ist Messfehler oder Falschmeldung.

**Wichtiger Nebeneffekt, den du kennen musst:** Weil `km` das laufende Maximum ist, hebt **ein einzelner zu hoher, aber unter 10.000 km liegender Ausreißer die gesamte Folgezeit des Fahrzeugs an**. Das ist eine bewusste Vereinfachung mit Risiko.

### 2.4 Bekannte Schwachstelle: Die Bereinigung läuft über die ganze Zeitreihe

`daily_series` wird einmal auf allen Daten ausgeführt, auch auf Messungen **nach** dem Stichtag. Für den Median in der Mitte eines 7er-Fensters werden dadurch bei Messungen kurz vor dem Stichtag auch Nachbarn nach dem Stichtag verwendet. Das ist eine **leichte Lücke** (Leakage) im Pseudo-Test. Ich habe nicht gemessen, wie groß der Effekt ist. Er ist vermutlich klein, aber nicht null.

Dazu kommt: Die Zielwerte im Pseudo-Test sind die **bereinigten** Stände. Die echte Testwahrheit ist vermutlich roh. Unsere Fehlerzahlen sind also eher **optimistisch**.

---

## 3. Der Pseudo-Test (`evaluate.py`)

### 3.1 Warum überhaupt ein Pseudo-Test?

Die Testdatei enthält die richtigen Antworten nicht. Ohne eigene Prüfung wüsstest du erst nach der Abgabe, ob es funktioniert. Der Pseudo-Test bildet die echte Aufgabe nach: einen Stichtag wählen, nur die Historie bis dahin nutzen, 133 Tage danach vorhersagen, mit den tatsächlichen Werten aus dem Sample vergleichen.

### 3.2 Horizont 133 Tage

Genau die Länge des echten Testfensters (02.07. bis 11.11.2026).

### 3.3 Stichtage 01.10.2025 und 01.02.2026

**Warum diese?**
- Das Ende des Fensters (Stichtag + 133 Tage) muss vor dem Ende der Daten (01.07.2026) liegen. Beide erfüllen das (Ende 11.02.2026 bzw. 14.06.2026).
- Zwei verschiedene Stichtage zeigen, ob das Ergebnis stabil ist.

**Schwäche (wichtig):** Der echte Test geht von **Juli bis November** (Sommer in den Herbst). Unsere Pseudo-Tests gehen von **Oktober bis Februar** und **Februar bis Juni**. Kilometerleistung ist saisonal. Es kann sein, dass das Modell im echten Zeitraum anders abschneidet. **Besser wäre ein zusätzlicher Stichtag 01.07.2025** (gleiche Jahreszeit, Ende 11.11.2025). Das ist machbar und sollte nachgeholt werden.

### 3.4 Was wird bewertet?

Alle tatsächlichen Messtage der Fahrzeuge im 133-Tage-Fenster. Für jede Messung wird `Vorhersage − echter Stand` berechnet. Das entspricht der echten Aufgabe, bei der die leeren Zeilen genau an solchen Tagen stehen.

---

## 4. Die Aufteilung der Daten (Splits)

### 4.1 Split nach Fahrzeugen: 80 % Training, 20 % Holdout

**Entscheidung:** Fahrzeuge werden einmal zufällig aufgeteilt (`split_vehicles`, Seed 0). 22.136 zum Lernen, 5.486 als Holdout.

**Warum nach Fahrzeug und nicht nach Zeilen?**
Ein Fahrzeug hat im Schnitt rund 98 Zeilen mit fast identischen statischen Merkmalen (Leistung, Hubraum, Maximalkilometer). Würde man Zeilen zufällig aufteilen, "kennt" das Modell jedes Testfahrzeug schon aus dem Training und könnte es über diese Merkmale wiedererkennen. Der Testfehler wäre unrealistisch niedrig. Diese Lücke heißt Leakage.

**Passt zur Aufgabe?** Ja, denn die README bewertet auf "held-out vehicles", und die echten Testfahrzeuge kommen im Sample gar nicht vor.

**Einschränkung:**
- Der Holdout stammt aus derselben Flotte und demselben Zeitraum wie das Training. Die echte Testflotte ist **jünger** (mittleres Baujahr ca. 2025,4 gegenüber 2023,9 im Sample) und zeitlich später. Der Holdout ist ein guter, aber kein perfekter Ersatz.
- Es wird nur **ein** Split verwendet. Eine Kreuzvalidierung (z. B. 5 Splits) wäre belastbarer.

### 4.2 Split nach Zeit: Training endet 133 Tage vor dem Prüf-Stichtag

**Entscheidung:** Trainingsstichtage liegen alle bei höchstens `Prüf-Stichtag − 133 Tage`.

**Warum?** Ein Trainingsbeispiel besteht aus Historie bis zum Stichtag C und dem Zuwachs in den 133 Tagen danach. Reicht das Ziel von C in die Prüfzeit hinein, hätte das Modell die Zukunft im Training gesehen. Mit dem 133-Tage-Abstand ist das ausgeschlossen.

**Training und Prüfung sind getrennt:** Für jeden Prüf-Stichtag wird ein eigenes Modell trainiert (nicht ein Modell für beide), damit der Zeitabstand sauber bleibt.

### 4.3 Training: 45-Tage-Raster ab 01.07.2022

**Warum 45 Tage?** Ein Kompromiss zwischen Menge an Lernbeispielen und Rechenzeit. Mit 45-Tage-Schritten kommen 24 bis 27 Stichtage zusammen, jeder liefert Beispiele von vielen Fahrzeugen.

**Warum ab 01.07.2022?** Das Sample beginnt im Januar 2022. Die ersten Monate dienen als Historie (für die Raten); ab Juli gibt es ausreichend Vorgeschichte.

**Schwäche:** Der Wert ist willkürlich und nicht optimiert.

### 4.4 Nur 8 Zeilen pro Fahrzeug und Stichtag im Training

**Warum?** Ohne Begrenzung würden Fahrzeuge mit vielen Messungen das Training dominieren (die Spanne reicht von 1 bis 366 Messungen pro Fahrzeug). Außerdem spart es Zeit und Speicher.

**Konsequenz:** Training und Bewertung sind unterschiedlich gewichtet. Im Training zählt jedes Fahrzeug höchstens 8-mal, in der Bewertung zählt jede Messung. Fahrzeuge mit vielen Messungen haben dort mehr Einfluss.

---

## 5. Die Baselines (`evaluate.py`)

### 5.1 Warum Baselines?

Die README bewertet "compared to a naive baseline". Und SIXT-Teams haben laut deiner Information selbst Baselines gebaut, mit denen deine Vorhersagen verglichen werden. Eine Fehlerzahl ohne Vergleich sagt nichts.

### 5.2 Die Rechnung

```
Vorhersage = letzter bekannter Stand + (km pro Tag) × (Tage seit der letzten Messung)
```

`gap` ist der Abstand zwischen Zieldatum und letzter bekannter Messung.

### 5.3 Die fünf Varianten

| Name | Rate | Idee |
|---|---|---|
| `b_last` | 0 | Untergrenze: "nichts passiert" |
| `b_rate_all` | Durchschnitt über die ganze Historie des Fahrzeugs | stabil, aber alt |
| `b_rate_30/90/180` | Durchschnitt der letzten 30/90/180 Tage | aktuell, aber verrauscht |
| `b_segment` | Median der Fahrzeugklasse (`vehicle_level`) | ignoriert das Einzelfahrzeug |

**Fallback:** Hat ein Fahrzeug keine eigene Rate (weniger als 14 Tage Spanne), wird die längere Rate oder die Segment-Rate verwendet.

### 5.4 Ergebnis und was daraus folgt

`b_segment` ist die beste Regel (MAE etwa 1.500 km). Das war überraschend: Fahrzeugspezifische Raten sind **schlechter** als eine Klassen-Rate. Der wahrscheinliche Grund: Die Rate eines einzelnen Fahrzeugs hängt stark von wenigen Mieten ab und schwankt. Das ist eine Vermutung, ich habe sie nicht separat geprüft.

**Folge:** Die Latte für ein Modell ist `b_segment`, nicht `b_last`. Gegen `b_last` zu gewinnen wäre zu einfach.

**Offen:** Ich weiß nicht, welche Baseline die SIXT-Teams benutzt haben. Wenn deren Baseline stärker ist als `b_segment`, ist unsere Latte zu niedrig.

---

## 6. Die Metriken (`metrics.py`)

| Metrik | Warum |
|---|---|
| **MAE** | Von der README genannt. Anschaulich in km. |
| **MedAE** | Zeigt den "typischen" Fehler ohne Einfluss weniger Riesenfehler. |
| **WAPE** | Prozentfehler, gewichtet. |
| **Bias** | Zeigt, ob das Modell systematisch zu hoch oder zu niedrig liegt. |

**Warum nicht MAPE?** Die README nennt MAPE. Bei kleinen echten Werten (Neufahrzeuge mit wenigen km, 1-%-Quantil liegt bei 11 km) explodiert der Prozentfehler. WAPE teilt die Summe der Fehler durch die Summe der echten Werte und ist dafür robust. Falls die Jury MAPE verlangt, kann man es zusätzlich berechnen.

**Horizont-Buckets** (0–14, 15–30, 31–60, 61–90, ab 91 Tage): Der Fehler wächst mit der Entfernung, und ein Durchschnitt über alles verdeckt das.

---

## 7. Das Modell (`model.py`)

### 7.1 Warum Gradient Boosting?

- Die Daten sind tabellarisch und gemischt (Zahlen und Kategorien, fehlende Werte). Das Verfahren (`HistGradientBoostingRegressor`) kommt damit ohne viel Aufbereitung klar.
- Es erfasst Zusammenhänge, die eine einzelne Rate nicht erfasst (z. B. Wechselwirkung von Klasse und Saison).
- Es trainiert auf rund 1 Mio. Zeilen in unter zwei Minuten.
- `scikit-learn` ist schon eine Abhängigkeit, es kommt nichts Neues hinzu.

**Alternativen, die ich nicht probiert habe:** LightGBM/XGBoost (ähnlich, evtl. schneller), lineare Modelle, neuronale Netze. Für einen Tag Arbeit schien Gradient Boosting der pragmatische Kandidat. Das ist eine **begründete Annahme, keine gemessene Überlegenheit**.

### 7.2 Was wird vorhergesagt? Der Zuwachs, nicht der Stand

**Ziel:** `km − letzter bekannter Stand`, mindestens 0.

**Warum?**
- Der absolute Stand reicht von 0 bis über 100.000 km. Ein Modell müsste erst lernen, dass der Stand beim letzten Wert beginnt.
- Als Zuwachs ist die Aufgabe klein und gut vergleichbar zwischen Fahrzeugen.
- Der Stand fällt nie, deshalb wird der Zuwachs auf mindestens 0 begrenzt.

### 7.3 Verlustfunktion: absoluter Fehler

**Entscheidung:** `loss="absolute_error"`.

**Warum?** Wir werden mit MAE bewertet. Wer den absoluten Fehler minimiert, optimiert direkt das Bewertungsmaß. Außerdem ist das Verfahren robuster gegen Ausreißer als der quadratische Fehler.

**Nebeneffekt:** Dieses Ziel schätzt den **Median** statt des Mittelwerts. Das erklärt vermutlich den negativen Bias (leicht zu niedrig geschätzt), denn die Verteilung des Zuwachses ist nach rechts schief. Das ist eine plausible Erklärung, keine geprüfte.

### 7.4 Die Features

| Gruppe | Features | Warum |
|---|---|---|
| Zeit | `gap`, `stale_days`, `target_month`, `target_dow` | Je größer der Abstand, desto größer der Zuwachs. Saison. |
| Fahrzeughistorie | `age_days`, `n_hist`, `last_km`, `km_share_of_max` | Alter, Datenmenge, Nutzungsgrad |
| Raten | `rate_all`, `rate_30/90/180`, `rate_fb` | Das beste Signal für künftiges Fahren |
| Erwartete Zuwächse | `exp_rate_*`, `exp_segment` (Baseline-Vorhersage minus letzter Stand) | Entscheidungsbäume können nicht multiplizieren. Mit dem fertigen Produkt `Rate × gap` als Feature muss das Modell es nicht erst lernen. |
| Fahrzeug | Klasse, Typ, Antrieb, Baujahr, Leistung, Hubraum, Maximalkilometer, ACRISS-Code | Unterschiedliche Fahrzeuge werden unterschiedlich genutzt |

**Bewusst nicht benutzt:** Alles, was in der Zukunft nicht bekannt ist, z. B. künftiger Mietstatus. Auch nicht verwendet: Branch- und Regionscodes, `uti_plan`, Produktanteil in der Historie. Das sind Kandidaten für Verbesserungen, nicht Fehler.

### 7.5 Hyperparameter

| Einstellung | Wert | Begründung |
|---|---|---|
| `learning_rate` | 0.08 | Üblicher Mittelwert |
| `max_iter` | 400 | Obergrenze für die Zahl der Bäume |
| `max_leaf_nodes` | 48 | Mittlere Baumgröße |
| `min_samples_leaf` | 100 | Verhindert zu kleine Gruppen |
| `early_stopping` | an | Bricht ab, wenn es nicht besser wird |

**Ehrlich gesagt:** Diese Werte sind **Standardwerte mit leichter Anpassung, nicht getunt**. Es wurde keine Suche gemacht. Dass das Modell alle 400 Iterationen genutzt hat, spricht dafür, dass es mehr Spielraum gibt.

**Schwäche des Early Stopping:** Es prüft auf einem zufälligen Zehntel der Trainingszeilen. Dieses enthält Zeilen derselben Fahrzeuge wie das Training und ist deshalb keine unabhängige Prüfung. Das Urteil darüber, ob eine Einstellung gut ist, muss der Holdout fällen.

---

## 8. Was die Ergebnisse bedeuten (und was nicht)

| Stichtag | `b_segment` (beste Regel) | `m_gboost` |
|---|---|---|
| 01.10.2025 | MAE 1.480 km, WAPE 6,5 % | MAE 1.348 km, WAPE 5,9 % |
| 01.02.2026 | MAE 1.497 km, WAPE 6,4 % | MAE 1.369 km, WAPE 5,8 % |

**Was wir sagen können:** Auf unbekannten Fahrzeugen schlägt das Modell die beste einfache Regel um etwa 9 % beim MAE.

**Was wir nicht sagen können:**
- Dass der Vorsprung auf der echten Testdatei so aussieht (andere Jahreszeit, jüngere Flotte, rohe Wahrheit statt bereinigter).
- Dass es das Modell mit der besten Einstellung ist (nicht getunt).
- Dass wir SIXTs Baseline schlagen (die kennen wir nicht).

---

## 9. Offene Punkte, nach Wichtigkeit

1. **Saisonaler Pseudo-Test** mit Stichtag 01.07.2025 ergänzen. Das ist die vergleichbarste Prüfung für den echten Test.
2. **Vorhersage auf `test_data.parquet`** bauen, damit eine lauffähige Abgabe existiert.
3. **Neustart-Fall:** 149 Test-Fahrzeuge haben gar keine Historie. Unsere Pipeline (innerer Join mit der Historie) lässt sie bisher einfach weg. Hier braucht es eine Rückfallregel (z. B. Segment-Rate ab 0).
4. **Tuning** (Iterationen, Baumgröße) mit Holdout-Fehler als Maßstab.
5. **Teil B** (Fehlererkennung) mit Flags aus Regeln und Residuen.
6. **Nowcast** klären (nötig oder nicht?).
7. **Fragen an SIXT:** Abgabeformat, Bedeutung der leeren `sixt_*`-Spalten, welche Baseline sie benutzt haben.
8. **Leakage durch die Bereinigung** (siehe 2.4) und die Bereinigung der Zielwerte prüfen.

---

## 10. Wie die Entscheidungen zur Bewertung passen

| Bewertungskriterium (README) | Unsere Antwort | Stärke |
|---|---|---|
| Fehler auf zurückgehaltenen **Fahrzeugen** | Fahrzeug-Split, Holdout nie im Training | stark |
| Fehler auf zurückgehaltenen **Zeiträumen** | Training endet 133 Tage vor dem Prüf-Stichtag, zwei Stichtage | mittel (Saison fehlt noch) |
| Vergleich mit **naiver Baseline** | Fünf Baselines, gleiche Zeilen wie das Modell | mittel (SIXTs Baseline unbekannt) |
| **Plausible Flags** (Teil B) | Nur Vorarbeit: `spike`, `decrease` | schwach, noch offen |
