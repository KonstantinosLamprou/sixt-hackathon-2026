"""Tuning des Forecast-Modells (Schritt 1 des Plans).

Ablauf:
  1. Fahrzeuge in drei Gruppen: Training 70 % / Validierung 15 % / Holdout 15 % (Holdout bleibt bis zum Schluss unberuehrt).
  2. Zufallssuche ueber Hyperparameter. Bewertet wird auf den VALIDIERUNGS-Fahrzeugen an den Stichtagen 2025-10-01 und
     2026-02-01. Der Stichtag 2025-07-01 (gleiche Jahreszeit wie der echte Test) wird NICHT zum Tuning benutzt.
  3. Bias-Korrektur: Skalierungsfaktor auf den vorhergesagten Zuwachs (Validierung).
  4. Ablation: zusaetzliche Status-Historie-Features, nur uebernommen bei >= 1 % besserem Validierungs-MAE.
  5. Endbewertung auf dem Holdout an allen drei Stichtagen, gegen Baselines und das ungetunte Modell.
  6. Permutations-Wichtigkeit der Features (Validierung).

Start (aus Project_2/):  python -m scripts.run_tuning
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import src.forecast as fc
import src.nowcast as nc
from src.baselines import HORIZON_DAYS
from src.config import FIGURES, MODELS, RESULTS
from src.data import SAMPLE, daily_series, load_raw, vehicle_static
from src.metrics import by_horizon

TUNE_CUTOFFS = ["2025-10-01", "2026-02-01"]
FINAL_CUTOFFS = ["2025-07-01", "2025-10-01", "2026-02-01"]
BASE_CFG = dict(learning_rate=0.08, max_iter=400, max_leaf_nodes=48, min_samples_leaf=100, l2_regularization=0.0,
                max_features=1.0)
SPACE = dict(learning_rate=[0.03, 0.05, 0.08, 0.12], max_iter=[300, 500, 800], max_leaf_nodes=[16, 31, 64, 128],
             min_samples_leaf=[50, 100, 300, 1000], l2_regularization=[0.0, 1.0, 10.0], max_features=[0.5, 0.8, 1.0])
N_RANDOM = 11
EXTRA_COLS = ["on_rent_share_30", "longterm_share_30", "on_rent_share_90", "longterm_share_90", "mean_rental_len_90",
              "uti_last", "n_rows_90"]


def split3(ids, val_frac=0.15, hold_frac=0.15, seed=0):
    """Deterministischer Fahrzeug-Split (train, val, holdout)."""
    ids = np.array(sorted(set(ids)))
    r = np.random.default_rng(seed).random(len(ids))
    return ids[r >= val_frac + hold_frac], ids[r < val_frac], ids[(r >= val_frac) & (r < val_frac + hold_frac)]


def status_history(series: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Status-Historie je Fahrzeug aus den letzten 30/90 Tagen vor dem Stichtag (nur Vergangenheit)."""
    c = pd.Timestamp(cutoff)
    h90 = series[(series.date <= c) & (series.date > c - pd.Timedelta(days=90))]
    g90 = h90.groupby("vhcl_anon_id").agg(on_rent_share_90=("on_rent", "mean"), longterm_share_90=("longterm", "mean"),
                                         mean_rental_len_90=("rental_len", "mean"), uti_last=("uti_plan", "last"),
                                         n_rows_90=("km", "size"))
    h30 = h90[h90.date > c - pd.Timedelta(days=30)]
    g30 = h30.groupby("vhcl_anon_id").agg(on_rent_share_30=("on_rent", "mean"), longterm_share_30=("longterm", "mean"))
    return g90.join(g30)


def add_extra(X, t, sh):
    return X.join(t[["vhcl_anon_id"]].join(sh, on="vhcl_anon_id")[EXTRA_COLS])


def build_train(series, static, cat_maps, ids, E, extra):
    cuts = pd.date_range(fc.TRAIN_START, pd.Timestamp(E) - pd.Timedelta(days=HORIZON_DAYS), freq=f"{fc.TRAIN_STEP_DAYS}D")
    Xs, ys = [], []
    for c in cuts:
        t, X = fc.make_dataset(series, static, c, ids, cat_maps, sample=True)
        if extra:
            X = add_extra(X, t, status_history(series, c))
        Xs.append(X)
        ys.append((t.km - t.last_km).clip(lower=0))
    return pd.concat(Xs), pd.concat(ys)


def build_eval(series, static, cat_maps, ids, E, extra):
    t, X = fc.make_dataset(series, static, E, ids, cat_maps)
    if extra:
        X = add_extra(X, t, status_history(series, E))
    return t, X


def fit(cfg, X, y):
    return HistGradientBoostingRegressor(loss="absolute_error", early_stopping=False, random_state=0, **cfg).fit(X, y)


def mae(model, t, X, alpha=1.0):
    pred = t.last_km + alpha * model.predict(X).clip(min=0)
    return float(np.abs(pred - t.km).mean())


def sample_configs(n, seed=1):
    rng = np.random.default_rng(seed)
    return [{k: v[rng.integers(len(v))] for k, v in SPACE.items()} for _ in range(n)]


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    raw = load_raw(SAMPLE)
    series = daily_series(raw).merge(nc.daily_status(raw), on=["vhcl_anon_id", "date"], how="left")
    static = vehicle_static(raw)
    cat_maps = {c: {v: i for i, v in enumerate(sorted(static[c].dropna().unique()))} for c in fc.CAT_COLS}
    train_ids, val_ids, hold_ids = split3(series.vhcl_anon_id.unique())
    print(f"Fahrzeuge: {len(train_ids)} Training, {len(val_ids)} Validierung, {len(hold_ids)} Holdout")

    # ---- 2. Zufallssuche ------------------------------------------------------------------------------------------
    data = {E: (*build_train(series, static, cat_maps, train_ids, E, False),
                *build_eval(series, static, cat_maps, val_ids, E, False)) for E in TUNE_CUTOFFS}
    log, best, best_models = [], (None, 1e18), None
    for i, cfg in enumerate([BASE_CFG] + sample_configs(N_RANDOM)):
        t0, maes, models = time.time(), {}, {}
        for E, (Xtr, ytr, tv, Xv) in data.items():
            models[E] = fit(cfg, Xtr, ytr)
            maes[E] = mae(models[E], tv, Xv)
        score = float(np.mean(list(maes.values())))
        log.append({"id": i, "base": i == 0, **cfg, **{f"val_mae_{E}": v for E, v in maes.items()},
                    "val_mae_mean": score, "seconds": round(time.time() - t0)})
        pd.DataFrame(log).to_csv(RESULTS / "tuning_forecast.csv", index=False)
        print(f"[{i:2d}] val MAE {score:7.1f}  ({time.time() - t0:3.0f}s)  {cfg}", flush=True)
        if score < best[1]:
            best, best_models = (cfg, score), models
    cfg, score = best
    base_score = log[0]["val_mae_mean"]
    print(f"\nBeste Konfiguration: {cfg}  val MAE {score:.1f}  (ungetunt: {base_score:.1f}, {100 * (1 - score / base_score):.1f} % besser)")

    # ---- 3. Bias-Korrektur ----------------------------------------------------------------------------------------
    alphas = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
    a_scores = {a: np.mean([mae(best_models[E], data[E][2], data[E][3], a) for E in TUNE_CUTOFFS]) for a in alphas}
    alpha = min(a_scores, key=a_scores.get)
    print("Bias-Faktor:", {a: round(s, 1) for a, s in a_scores.items()}, "->", alpha)

    # ---- 4. Ablation: Status-Historie-Features --------------------------------------------------------------------
    maes_x = {}
    data_x = {}
    for E in TUNE_CUTOFFS:
        Xtr, ytr = build_train(series, static, cat_maps, train_ids, E, True)
        tv, Xv = build_eval(series, static, cat_maps, val_ids, E, True)
        data_x[E] = (Xtr, ytr, tv, Xv)
        maes_x[E] = mae(fit(cfg, Xtr, ytr), tv, Xv, alpha)
    score_x = float(np.mean(list(maes_x.values())))
    score_a = float(a_scores[alpha])
    use_extra = score_x < 0.99 * score_a
    print(f"Status-Historie-Features: val MAE {score_x:.1f} gegen {score_a:.1f} ohne -> {'UEBERNOMMEN' if use_extra else 'verworfen'}")
    json.dump({"cfg": cfg, "alpha": alpha, "use_extra_features": use_extra,
               "val_mae_used": score_x if use_extra else score_a, "val_mae_without_extra": score_a,
               "val_mae_with_extra": score_x, "val_mae_untuned_cfg": base_score},
              open(RESULTS / "best_forecast_params.json", "w"), indent=2)

    # ---- 5. Endbewertung auf dem Holdout --------------------------------------------------------------------------
    out = []
    for E in FINAL_CUTOFFS:
        Xtr, ytr = build_train(series, static, cat_maps, train_ids, E, use_extra)
        base_cols = [c for c in Xtr.columns if c not in EXTRA_COLS]
        m_base = fit(BASE_CFG, Xtr[base_cols], ytr)
        m_tuned = fit(cfg, Xtr, ytr)
        t, X = build_eval(series, static, cat_maps, hold_ids, E, use_extra)
        t["m_untuned"] = t.last_km + m_base.predict(X[base_cols]).clip(min=0)
        t["m_tuned"] = t.last_km + alpha * m_tuned.predict(X).clip(min=0)
        res = by_horizon(t, ["b_last", "b_segment", "m_untuned", "m_tuned"])
        res.insert(0, "cutoff", E)
        out.append(res)
        print(f"\n=== Holdout, Stichtag {E}: {t.vhcl_anon_id.nunique()} Fahrzeuge, {len(t)} Zeilen ===")
        print(res[res.horizon == "all"].round(1).to_string(index=False), flush=True)
        if E == FINAL_CUTOFFS[-1]:
            joblib.dump({"model": m_tuned, "alpha": alpha, "columns": list(Xtr.columns), "cfg": cfg},
                        MODELS / f"forecast_tuned_{E}.joblib", compress=3)
    pd.concat(out).to_csv(RESULTS / "forecast_tuned_holdout.csv", index=False)

    # ---- 6. Feature-Wichtigkeit (Validierung, letzter Stichtag) ---------------------------------------------------
    E = FINAL_CUTOFFS[-1]
    tv, Xv = build_eval(series, static, cat_maps, val_ids, E, use_extra)
    s = tv.sample(n=min(20_000, len(tv)), random_state=0)
    y = (s.km - s.last_km).clip(lower=0)
    pi = permutation_importance(m_tuned, Xv.loc[s.index], y, scoring="neg_mean_absolute_error", n_repeats=3,
                                random_state=0, n_jobs=-1)
    imp = pd.DataFrame({"feature": Xv.columns, "mae_increase": pi.importances_mean}).sort_values("mae_increase", ascending=False)
    imp.to_csv(RESULTS / "forecast_feature_importance.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = imp.head(20).iloc[::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(top.feature, top.mae_increase)
    plt.xlabel("Anstieg des MAE (km) wenn das Feature durchgemischt wird")
    plt.title(f"Forecast: Permutations-Wichtigkeit (Validierung, Stichtag {E})")
    plt.tight_layout()
    plt.savefig(FIGURES / "forecast_feature_importance.png", dpi=150)
    print("\nTop-10 Features:\n", imp.head(10).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
