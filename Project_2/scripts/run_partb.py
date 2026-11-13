"""Teil B: Schwellen auf dem Sample bestimmen, mit eingespritzten Fehlern pruefen, auf die bekannten Testzeilen anwenden.

Start (aus Project_2/):  python -m scripts.run_partb
Ausgaben: results/partb_summary.csv, results/partb_injection.csv, data/processed/partb_flagged_rows.csv (nur markierte Zeilen),
          data/interim/partb_flags_all_rows.parquet (alle bekannten Testzeilen mit flag 0/1, nicht in Git)
"""
import numpy as np
import pandas as pd

from src import anomaly as an
from src.config import INTERIM, PROCESSED, RESULTS
from src.data import SAMPLE, TEST, load_raw, vehicle_static


def level_of(df):
    return vehicle_static(df)["vehicle_level"]


def injection_test(d, level, seg_rate, thr, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    d = d.reset_index(drop=True)
    first = d.groupby("vhcl_anon_id").cumcount() == 0
    pos = rng.choice(d.index[~first & (d.km > 2000)], size=n, replace=False)
    kinds = np.array(["x10", "div10", "zero", "plus_5k_30k", "minus_20_60pct", "minus_10k_30k"])
    kind = kinds[rng.integers(len(kinds), size=n)]
    inj = d.copy()
    km = inj.km.values.astype(float)
    for p, k in zip(pos, kind):
        v = km[p]
        km[p] = {"x10": v * 10, "div10": v // 10, "zero": 0, "plus_5k_30k": v + rng.integers(5000, 30000),
                 "minus_20_60pct": v * rng.uniform(0.4, 0.8), "minus_10k_30k": max(v - rng.integers(10000, 30000), 0)}[k]
    inj["km"] = km
    res = an.detect(inj, level, seg_rate, thr)
    hit = res.loc[pos, "flag"].values
    near = set(pos) | set(pos + 1)                       # Nachbarmessung zaehlt als Begleitschaden, nicht als Fehlalarm
    mask = np.array([i not in near for i in range(len(res))])
    fp = res.loc[mask, "flag"].mean()
    out = pd.DataFrame({"error_type": kind, "detected": hit}).groupby("error_type").detected.agg(["size", "mean"]).rename(
        columns={"size": "n", "mean": "recall"})
    out.loc["ALL"] = [n, hit.mean()]
    return out, fp


def main():
    RESULTS.mkdir(exist_ok=True), PROCESSED.mkdir(parents=True, exist_ok=True), INTERIM.mkdir(parents=True, exist_ok=True)
    raw = load_raw(SAMPLE)
    d_s, lvl_s = an.daily_readings(raw), level_of(raw)
    seg_rate, thr = an.fit(d_s, lvl_s)
    print("Segment-Raten (km/Tag):", {k: round(v, 1) for k, v in seg_rate.items()})
    print("Schwellen Zuwachs-Abweichung (km) je Zeitabstand-Klasse:\n hoch", {k: round(v) for k, v in thr["hi"].items()},
          "\n tief", {k: round(v) for k, v in thr["lo"].items()})
    res_s = an.detect(d_s, lvl_s, seg_rate, thr)
    rates = res_s[an.FLAGS + ["flag"]].mean() * 100
    print(f"\nSample: {len(res_s)} Meldungen, Flag-Anteile in %:\n{rates.round(3).to_string()}")

    inj, fp = injection_test(d_s, lvl_s, seg_rate, thr)
    print(f"\nEingespritzte Fehler (6000): Recall je Art; Fehlalarmrate auf unberuehrten Meldungen: {fp * 100:.2f} %")
    print(inj.round(3).to_string())
    inj.to_csv(RESULTS / "partb_injection.csv")

    raw_t = load_raw(TEST)
    d_t, lvl_t = an.daily_readings(raw_t), level_of(raw_t)
    res_t = an.detect(d_t, lvl_t, seg_rate, thr)
    res_t["reason"] = an.reasons(res_t)
    rt = res_t[an.FLAGS + ["flag"]].mean() * 100
    print(f"\nTest: {len(res_t)} Meldungen von {res_t.vhcl_anon_id.nunique()} Fahrzeugen, Flag-Anteile in %:\n{rt.round(3).to_string()}")
    pd.DataFrame({"sample_%": rates, "test_%": rt}).to_csv(RESULTS / "partb_summary.csv")

    # auf die Original-Zeilen der Testdatei zurueckschreiben: Zeile mit dem gemeldeten Stand dieses Tages
    rows = raw_t.dropna(subset=["vhcl_current_mileage"])[["vhcl_anon_id", "vhcl_fct_taken_datm", "date", "vhcl_current_mileage"]]
    m = rows.merge(res_t[["vhcl_anon_id", "date", "km", "flag", "score", "reason"] + an.FLAGS], on=["vhcl_anon_id", "date"], how="left")
    m["flag"] = ((m.flag == 1) & (m.vhcl_current_mileage == m.km)).astype(int)
    m.loc[m.flag == 0, ["score", "reason"]] = [0.0, ""]
    m = m.drop(columns=["date", "km"])
    m.to_parquet(INTERIM / "partb_flags_all_rows.parquet", index=False)
    fl = m[m.flag == 1].sort_values("score", ascending=False)
    fl.to_csv(PROCESSED / "partb_flagged_rows.csv", index=False)
    print(f"\nTestzeilen mit gemeldetem Stand: {len(m)}, geflaggt: {len(fl)} ({100 * len(fl) / len(m):.2f} %)")
    print(fl[["vhcl_fct_taken_datm", "vhcl_current_mileage", "score", "reason"]].head(8).to_string(index=False, max_colwidth=90))


if __name__ == "__main__":
    main()
