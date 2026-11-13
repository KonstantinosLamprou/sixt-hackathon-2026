"""Builds submission/slides.pdf (5 slides) from the result tables in results/.  Start:  python -m scripts.make_slides"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

from src.config import FIGURES, RESULTS, ROOT

INK, MUTED, ACCENT, GOOD, GREY = "#1b2430", "#5b6675", "#e8590c", "#2b8a3e", "#adb5bd"
W, H = 13.33, 7.5
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def page(title, subtitle=None):
    fig = plt.figure(figsize=(W, H))
    fig.text(0.05, 0.92, title, fontsize=26, fontweight="bold", color=INK, va="center")
    if subtitle:
        fig.text(0.05, 0.865, subtitle, fontsize=13.5, color=MUTED, va="center")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.835, 0.835], color=ACCENT, lw=2.5))
    return fig


def tile(fig, x, y, w, h, big, label, sub=None, color=ACCENT):
    ax = fig.add_axes([x, y, w, h]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04", fc="#f4f6f8", ec="none", transform=ax.transAxes))
    ax.text(0.06, 0.74, big, fontsize=27, fontweight="bold", color=color, va="center", transform=ax.transAxes)
    ax.text(0.06, 0.45, label, fontsize=12.5, fontweight="bold", color=INK, va="center", transform=ax.transAxes)
    if sub:
        ax.text(0.06, 0.2, sub, fontsize=10.5, color=MUTED, va="center", transform=ax.transAxes, linespacing=1.35)


def bullets(fig, x, y, lines, size=12.5, gap=0.052):
    for i, t in enumerate(lines):
        fig.text(x, y - i * gap, t, fontsize=size, color=INK, va="top")


def rd(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    hold, nc, cs, inj, summ, bt = (rd("forecast_tuned_holdout.csv"), rd("nowcast_metrics.csv"), rd("coldstart_metrics.csv"),
                                   rd("partb_injection.csv"), rd("partb_summary.csv"), rd("backtest_test_fleet.csv"))
    cut_lab = {"2025-07-01": "1 Jul 2025\n(same season as test)", "2025-10-01": "1 Oct 2025", "2026-02-01": "1 Feb 2026"}
    allh = hold[hold.horizon == "all"].pivot(index="cutoff", columns="model", values="MAE")
    ncall = nc[nc.horizon == "all"].pivot(index="cutoff", columns="model", values="MAE")
    best_rule_gain = (1 - allh["m_tuned"] / allh["b_segment"]) * 100
    nc_gain = (1 - ncall["m_nc_status"] / ncall["b_segment"]) * 100
    rec = inj.set_index(inj.columns[0])["recall"]
    flagged = int(round(summ.loc[summ.iloc[:, 0] == "flag", "test_%"].iloc[0] / 100 * 493554)) if summ is not None else 5338

    pdf = PdfPages(ROOT / "submission" / "slides.pdf")

    # ---- 1 headline -------------------------------------------------------------------------------------------------------
    f = page("Mileage forecast + erroneous-report flags", "One prediction table for all 656,601 test rows: forecast, cold start and error flags")
    if bt is not None:
        b = bt[bt.horizon == "all"].set_index("model").MAE
        mdl = "m_sample+test_iso" if "m_sample+test_iso" in b.index else b.index[-1]
        tile(f, 0.05, 0.50, 0.28, 0.27, f"{b[mdl]:,.0f} km", "Forecast MAE, 133 days",
             f"backtest on the real test fleet\nvs {b['b_segment']:,.0f} km best rule, {b['b_last']:,.0f} km naive")
    else:
        tile(f, 0.05, 0.50, 0.28, 0.27, f"{allh['m_tuned'].mean():,.0f} km", "Forecast MAE, 133 days", "held-out vehicles, 3 cut-off dates")
    tile(f, 0.36, 0.50, 0.28, 0.27, f"{ncall['m_nc_status'].min():,.0f}-{ncall['m_nc_status'].max():,.0f} km", "Nowcast MAE, up to 28 days",
         f"{nc_gain.min():.0f}-{nc_gain.max():.0f}% better than the best rule", color=GOOD)
    tile(f, 0.67, 0.50, 0.28, 0.27, f"{rec['ALL'] * 100:.1f}%", "Recall on injected errors", f"{flagged:,} of 493,554 reports flagged (1.1%)", color=GOOD)
    tile(f, 0.05, 0.19, 0.28, 0.25, "163,047", "forecast rows filled", "159,302 model + 3,745 cold start\n(149 new cars without history)", color=INK)
    tile(f, 0.36, 0.19, 0.28, 0.25, "0 / 1", "error flag per reported mileage", "rule violations + deviation from\nexpected usage, with score and reason", color=INK)
    tile(f, 0.67, 0.19, 0.28, 0.25, "1 file", "submission/predictions.parquet", "same row order as test_data.parquet,\nkey: row_id", color=INK)
    fig_note = "All accuracy numbers are measured on vehicles held out from training and tuning (no labels exist for the hidden test period)."
    f.text(0.05, 0.1, fig_note, fontsize=11, color=MUTED)
    pdf.savefig(f); plt.close(f)

    # ---- 2 forecast -------------------------------------------------------------------------------------------------------
    f = page("Forecast: 133 days ahead", "Gradient boosting on the mileage increment since the last known reading; error in km (MAE), lower is better")
    ax = f.add_axes([0.07, 0.14, 0.50, 0.60])
    x = np.arange(len(allh)); w = 0.26
    for i, (col, lab, c) in enumerate([("b_last", "Naive: last value", GREY), ("b_segment", "Best rule: class median rate x days", MUTED), ("m_tuned", "Model", ACCENT)]):
        bars = ax.bar(x + (i - 1) * w, allh[col], w, label=lab, color=c)
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in allh[col]], fontsize=9.5, padding=2)
    ax.set_xticks(x, [cut_lab[c] for c in allh.index], fontsize=10.5); ax.set_ylabel("MAE (km)"); ax.legend(frameon=False, fontsize=10.5)
    ax.set_title("Held-out vehicles, 3 cut-off dates", fontsize=12.5, color=INK, loc="left")
    y0 = 0.74
    f.text(0.62, y0, "Model beats the best rule by", fontsize=13, color=MUTED)
    for i, (c, g) in enumerate(best_rule_gain.items()):
        f.text(0.62, y0 - 0.065 * (i + 1), f"{g:.1f}%", fontsize=20, fontweight="bold", color=GOOD if g > 8 else ACCENT)
        f.text(0.72, y0 - 0.065 * (i + 1) + 0.004, c, fontsize=12, color=INK)
    if bt is not None:
        f.text(0.62, 0.40, "Backtest on the real test fleet", fontsize=13, fontweight="bold", color=INK)
        f.text(0.62, 0.365, "forecast from 18 Feb 2026, compared with the readings\nknown until 1 Jul 2026 (same 133-day setup)", fontsize=10.5, color=MUTED, va="top")
        rows = bt[bt.horizon == "all"].set_index("model").MAE
        names = {"b_last": "Naive (last value)", "b_segment": "Best rule", "m_sample": "Model", "m_sample+test": "Model + test history",
                 "m_sample+test_iso": "Model + test history, monotone"}
        for i, (k, lab) in enumerate(names.items()):
            if k in rows.index:
                f.text(0.62, 0.275 - i * 0.045, f"{lab}", fontsize=11, color=INK)
                f.text(0.92, 0.275 - i * 0.045, f"{rows[k]:,.0f}", fontsize=11, color=INK, fontweight="bold", ha="right")
    pdf.savefig(f); plt.close(f)

    # ---- 3 nowcast + cold start -------------------------------------------------------------------------------------------
    f = page("Nowcast and new vehicles", "Nowcast: mileage on a day without report (up to 28 days), using the rental status of that day")
    ax = f.add_axes([0.07, 0.14, 0.45, 0.60])
    x = np.arange(len(ncall))
    for i, (col, lab, c) in enumerate([("b_segment", "Best rule", MUTED), ("m_nc_history", "Model, history only", "#f59f00"), ("m_nc_status", "Model + rental status", ACCENT)]):
        bars = ax.bar(x + (i - 1) * w, ncall[col], w, label=lab, color=c)
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in ncall[col]], fontsize=9.5, padding=2)
    ax.set_xticks(x, [cut_lab[c] for c in ncall.index], fontsize=10.5); ax.set_ylabel("MAE (km)"); ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    ax.set_ylim(0, ncall.values.max() * 1.18)
    f.text(0.60, 0.74, "Cold start: 149 new cars with no mileage history", fontsize=14, fontweight="bold", color=INK)
    f.text(0.60, 0.70, "start at a typical new-car reading (about 12 km) and follow the\nmedian growth curve of new vehicles by age in days", fontsize=11, color=MUTED, va="top")
    piv = cs.pivot(index="cutoff", columns="rule", values="MAE")
    f.text(0.60, 0.545, "MAE on simulated new cars (km)", fontsize=11.5, color=INK, fontweight="bold")
    f.text(0.60, 0.51, "cut-off", fontsize=10.5, color=MUTED); f.text(0.74, 0.51, "fleet rate", fontsize=10.5, color=MUTED, ha="right")
    f.text(0.84, 0.51, "age curve", fontsize=10.5, color=MUTED, ha="right")
    for i, (c, r) in enumerate(piv.iterrows()):
        f.text(0.60, 0.47 - i * 0.045, c, fontsize=11, color=INK)
        f.text(0.74, 0.47 - i * 0.045, f"{r['fleet_rate']:,.0f}", fontsize=11, color=INK, ha="right")
        f.text(0.84, 0.47 - i * 0.045, f"{r['age_curve']:,.0f}", fontsize=11, color=ACCENT, fontweight="bold", ha="right")
    f.text(0.60, 0.30, "The age curve is never worse than the fleet-rate rule\nand cuts the error by 15% at one cut-off.", fontsize=11, color=MUTED, va="top")
    pdf.savefig(f); plt.close(f)

    # ---- 4 anomaly flags --------------------------------------------------------------------------------------------------
    f = page("Erroneous-report flags", "A report is flagged when it breaks physics or deviates strongly from expected usage")
    ax = f.add_axes([0.07, 0.14, 0.42, 0.58])
    order = [k for k in rec.index if k != "ALL"]
    vals = [rec[k] * 100 for k in order]
    names = {"x10": "digit appended (x10)", "div10": "digit dropped (/10)", "zero": "reset to 0", "plus_5k_30k": "+5,000 to +30,000 km",
             "minus_20_60pct": "-20% to -60%", "minus_10k_30k": "-10,000 to -30,000 km"}
    bars = ax.barh([names.get(k, k) for k in order][::-1], vals[::-1], color=GOOD)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in vals[::-1]], fontsize=10.5, padding=3)
    ax.set_xlim(90, 102); ax.set_xlabel("recall on 6,000 injected errors (%)")
    ax.set_title(f"Detected: {rec['ALL'] * 100:.1f}% overall", fontsize=12.5, color=INK, loc="left")
    f.text(0.55, 0.74, "Five flags, combined with OR", fontsize=14, fontweight="bold", color=INK)
    bullets(f, 0.55, 0.69, ["rollback: reading below the previous one (longest",
                            "   non-decreasing subsequence marks the outlier itself)",
                            "too high: above all later readings",
                            "speed: more than 1,500 km/day between readings",
                            "high / low: increase far above or below the expected",
                            "   usage for that time gap (quantile bands per gap)"], size=11.5, gap=0.042)
    f.text(0.55, 0.40, "On the test file", fontsize=14, fontweight="bold", color=INK)
    bullets(f, 0.55, 0.355, [f"{flagged:,} of 493,554 reports flagged (1.1%)",
                             "2,781 by rules only, 2,257 by deviation only, 300 by both",
                             "every flag has a score and a plain-text reason"], size=11.5, gap=0.042)
    f.text(0.05, 0.075, "Injected errors are large; subtle manipulation is not covered by this test. No ground-truth labels exist.", fontsize=10.5, color=MUTED)
    pdf.savefig(f); plt.close(f)

    # ---- 5 deliverables and limits ----------------------------------------------------------------------------------------
    f = page("Deliverables, limits, next steps")
    f.text(0.05, 0.78, "What is in the repo", fontsize=14, fontweight="bold", color=INK)
    bullets(f, 0.05, 0.735, ["submission/predictions.parquet: forecast + error flag for every test row",
                             "models/: final forecast model and nowcast model (joblib)",
                             "results/: all evaluation tables; reports/figures/: feature importance",
                             "src/ + scripts/: full pipeline, run with python -m scripts.run_<step>",
                             "SUBMISSION.md: approach, results, how to run"], size=11.5, gap=0.042)
    f.text(0.05, 0.47, "Known limits", fontsize=14, fontweight="bold", color=INK)
    bullets(f, 0.05, 0.425, ["Model underestimates in the summer cut-off (bias -516 km): no seasonal feature yet",
                             "Gain over the best rule varies by holdout and date: 5% to 12%",
                             "Error flags use a simple expected rate, not the trained model; no labels to verify precision",
                             "Nowcast is not filled per test row (empty test rows have no rental status)"], size=11.5, gap=0.042)
    f.text(0.55, 0.78, "Next steps", fontsize=14, fontweight="bold", color=INK)
    bullets(f, 0.55, 0.735, ["Seasonal index and fleet trend as forecast features",
                             "Out-of-fold nowcast as expected value for the flags,",
                             "   with quantile uncertainty bands",
                             "Cross-validation over vehicles",
                             "Checks with subtle manipulation (a few percent)"], size=11.5, gap=0.042)
    pdf.savefig(f); plt.close(f)
    pdf.close()
    print("saved submission/slides.pdf")


if __name__ == "__main__":
    main()
