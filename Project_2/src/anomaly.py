"""Teil B (einfache Form): fehlerhafte Kilometermeldungen erkennen.

Eine "Meldung" = hoechster Stand eines Fahrzeugs an einem Tag. Fuenf Flags:
  f_rollback : Stand liegt unter dem Stand davor, obwohl spaetere Messungen darueber liegen (laengste nicht fallende
               Teilfolge; markiert den Ausreisser selbst, nicht die Kaskade danach)
  f_toohigh  : Stand liegt ueber allen spaeteren Messungen (zu hoch gemeldet)
  f_speed    : mehr als MAX_KM_PER_DAY km pro Tag zwischen zwei plausiblen Messungen
  f_high     : Zuwachs deutlich ueber dem Erwartbaren (Fahrzeug-Median-Rate x Tage) fuer diesen Zeitabstand
  f_low      : Zuwachs deutlich unter dem Erwartbaren (moegliche Untermeldung)
Die Schwellen fuer f_high/f_low sind Quantile der Abweichungen je Zeitabstand, bestimmt auf dem Sample (nicht auf den
Testdaten), und werden auf die Testdaten angewendet.
"""
import bisect

import numpy as np
import pandas as pd

TOL_KM = 50
MAX_KM_PER_DAY = 1500
Q_HIGH, Q_LOW = 0.995, 0.002
BIN_EDGES = [0, 1, 2, 4, 7, 14, 28, 1e9]
FLAGS = ["f_rollback", "f_toohigh", "f_speed", "f_high", "f_low"]


def daily_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Hoechster gemeldeter Stand je Fahrzeug und Tag (df braucht Spalte 'date')."""
    d = (df.dropna(subset=["vhcl_current_mileage"]).groupby(["vhcl_anon_id", "date"])["vhcl_current_mileage"]
         .max().rename("km").reset_index())
    return d.sort_values(["vhcl_anon_id", "date"]).reset_index(drop=True)


def _lnds_mask(a: np.ndarray) -> np.ndarray:
    """Maske der laengsten nicht fallenden Teilfolge."""
    n, tails, tails_idx, prev = len(a), [], [], [-1] * len(a)
    for i, x in enumerate(a):
        j = bisect.bisect_right(tails, x)
        if j == len(tails):
            tails.append(x), tails_idx.append(i)
        else:
            tails[j], tails_idx[j] = x, i
        prev[i] = tails_idx[j - 1] if j > 0 else -1
    mask, k = np.zeros(n, bool), tails_idx[-1]
    while k != -1:
        mask[k] = True
        k = prev[k]
    return mask


def _prepare(d: pd.DataFrame, seg_rate: dict, level: pd.Series):
    d = d.copy()
    ids, km = d.vhcl_anon_id.values, d.km.values
    bounds = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
    kept = np.zeros(len(d), bool)
    for s, e in zip(bounds[:-1], bounds[1:]):
        kept[s:e] = _lnds_mask(km[s:e])
    d["kept"] = kept
    kv = d.km.where(d.kept)
    d["lo"], d["hi"] = kv.groupby(d.vhcl_anon_id).ffill(), kv.groupby(d.vhcl_anon_id).bfill()
    d["f_rollback"] = (~d.kept) & (d.km < d.lo - TOL_KM)
    d["f_toohigh"] = (~d.kept) & (d.km > d.hi + TOL_KM)

    k = d[d.kept].copy()
    gk = k.groupby("vhcl_anon_id")
    k["dkm"], k["dd"] = gk.km.diff(), gk.date.diff().dt.days.clip(lower=1)
    k["rate"] = k.dkm / k.dd
    g = k.groupby("vhcl_anon_id").rate
    ref_own = g.transform("median").where(g.transform("count") >= 5)
    glob = float(np.nanmedian(list(seg_rate.values())))
    k["ref"] = ref_own.fillna(k.vhcl_anon_id.map(level).map(seg_rate)).fillna(glob)
    k["excess"] = k.dkm - k.ref * k.dd
    k["bin"] = pd.cut(k.dd, bins=BIN_EDGES, labels=False)
    k["f_speed"] = k.rate > MAX_KM_PER_DAY
    return d, k


def fit(d: pd.DataFrame, level: pd.Series):
    """Bestimmt Segment-Raten und Schwellen je Zeitabstand auf dem uebergebenen Datensatz (hier: Sample)."""
    d0 = d.copy()
    ids, km = d0.vhcl_anon_id.values, d0.km.values
    bounds = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
    kept = np.zeros(len(d0), bool)
    for s, e in zip(bounds[:-1], bounds[1:]):
        kept[s:e] = _lnds_mask(km[s:e])
    k = d0[kept].copy()
    gk = k.groupby("vhcl_anon_id")
    k["rate"] = gk.km.diff() / gk.date.diff().dt.days.clip(lower=1)
    veh = k.groupby("vhcl_anon_id").rate.median()
    seg_rate = veh.groupby(level.reindex(veh.index)).median().to_dict()
    _, kk = _prepare(d, seg_rate, level)
    q = kk.dropna(subset=["excess"]).groupby("bin").excess
    thr = {"hi": q.quantile(Q_HIGH).to_dict(), "lo": q.quantile(Q_LOW).to_dict()}
    return seg_rate, thr


def detect(d: pd.DataFrame, level: pd.Series, seg_rate: dict, thr: dict) -> pd.DataFrame:
    d, k = _prepare(d, seg_rate, level)
    hi, lo = k["bin"].map(thr["hi"]), k["bin"].map(thr["lo"])
    k["f_high"], k["f_low"] = k.excess > hi, k.excess < lo
    k["score"] = np.where(k.excess > 0, k.excess / hi, k.excess / lo)
    d = d.join(k[["dkm", "dd", "ref", "f_speed", "f_high", "f_low", "score"]])
    for f in FLAGS:
        d[f] = d[f].fillna(False).astype(bool)
    d["flag"] = d[FLAGS].any(axis=1).astype(int)
    d["score"] = d["score"].abs().fillna(0.0)
    d.loc[d.f_rollback | d.f_toohigh | d.f_speed, "score"] = d.score.clip(lower=1.0) * 2
    return d


def _reason(r) -> str:
    out = []
    if r.f_rollback:
        out.append(f"Reading {r.km:.0f} km is below the previous reading ({r.lo:.0f} km)")
    if r.f_toohigh:
        out.append(f"Reading {r.km:.0f} km is above all later readings ({r.hi:.0f} km)")
    if r.f_speed:
        out.append(f"more than {MAX_KM_PER_DAY} km/day (increase of {r.dkm:.0f} km in {r.dd:.0f} days)")
    if r.f_high:
        out.append(f"increase of {r.dkm:.0f} km in {r.dd:.0f} days, expected about {r.ref * r.dd:.0f} km")
    if r.f_low:
        out.append(f"increase of only {r.dkm:.0f} km in {r.dd:.0f} days, expected about {r.ref * r.dd:.0f} km")
    return "; ".join(out)


def reasons(d: pd.DataFrame) -> pd.Series:
    """Begruendungstext je geflaggter Meldung (leer bei flag == 0)."""
    out = pd.Series("", index=d.index)
    f = d[d.flag == 1]
    out.loc[f.index] = [_reason(r) for r in f.itertuples()]
    return out
