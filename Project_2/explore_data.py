"""Datenueberblick fuer Project 2. Usage: python explore_data.py  -> schreibt data_overview.txt"""
import io
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FILES = ["data_30_percent_sample.parquet", "test_data.parquet"]
out = io.StringIO()


def p(*a):
    print(*a, file=out)


pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 40)
dfs = {}
for f in FILES:
    df = pd.read_parquet(HERE / "data" / f)
    dfs[f] = df
    p("=" * 100, f"\nDATEI: {f}  shape={df.shape}  RAM={df.memory_usage(deep=True).sum()/1e6:.0f} MB\n", "=" * 100)
    p("\n-- dtypes / fehlend / eindeutige Werte --")
    p(pd.DataFrame({"dtype": df.dtypes.astype(str), "missing_%": (df.isna().mean() * 100).round(2),
                    "nunique": df.nunique()}).to_string())
    p("\n-- head(8) --"); p(df.head(8).T.to_string())
    num = df.select_dtypes("number")
    if len(num.columns):
        p("\n-- numerische Statistik --"); p(num.describe(percentiles=[.01, .5, .99]).T.to_string())
    p("\n-- Kategorien (Haeufigkeit, max 12) --")
    for c in df.columns:
        if df[c].nunique() <= 15 and c != "vhcl_anon_id":
            p(f"[{c}]"); p(df[c].value_counts(dropna=False).head(12).to_string())
    if "vhcl_anon_id" in df:
        t = pd.to_datetime(df["vhcl_fct_taken_datm"])
        n = df.groupby("vhcl_anon_id").size()
        p("\n-- Fahrzeuge --")
        p(f"Fahrzeuge: {n.size}, Zeitraum: {t.min()} bis {t.max()}")
        p("Messungen pro Fahrzeug:"); p(n.describe().to_string())
        p(f"Exakte Duplikatzeilen: {df.duplicated().sum()}")
        p(f"Zeilen ohne Mileage: {df['vhcl_current_mileage'].isna().sum()}")
        d = df.assign(t=t).dropna(subset=["vhcl_current_mileage"]).sort_values(["vhcl_anon_id", "t", "vhcl_current_mileage"])
        g = d.groupby("vhcl_anon_id")
        diff = g.vhcl_current_mileage.diff()
        days = g.t.diff().dt.days
        p(f"Mileage faellt (diff<0, bei Sortierung nach Tag+Stand): {(diff < 0).sum()}")
        p("Tage zwischen Messungen:"); p(days.describe(percentiles=[.5, .9, .99]).to_string())
        rate = diff / days.where(days > 0)
        p("km/Tag (nur days>0):"); p(rate.describe(percentiles=[.01, .5, .99, .999]).to_string())
        p(f"Rate > 1000 km/Tag: {(rate > 1000).sum()}")
        p("\n-- 1 Beispielfahrzeug (erste 25 Zeilen) --")
        vid = n.idxmax()
        c = [x for x in ["vhcl_fct_taken_datm", "vhcl_current_mileage", "status_code_anonymized", "on_rent_status",
                         "rental_product", "rntl_handover_date", "rntl_return_date", "vhcl_max_mileage", "uti_plan"] if x in df]
        p(df[df.vhcl_anon_id == vid].sort_values("vhcl_fct_taken_datm")[c].head(25).to_string())
    p()

if len(dfs) == 2:
    a, b = dfs.values()
    p("=" * 100, "\nVERGLEICH sample vs. test\n", "=" * 100)
    p("Spalten nur in sample:", set(a.columns) - set(b.columns))
    p("Spalten nur in test:", set(b.columns) - set(a.columns))
    if "vhcl_anon_id" in a and "vhcl_anon_id" in b:
        s = set(a.vhcl_anon_id)
        p(f"Fahrzeuge in test: {b.vhcl_anon_id.nunique()}, davon auch in sample: {len(s & set(b.vhcl_anon_id))}")

(HERE / "data_overview.txt").write_text(out.getvalue())
print("geschrieben:", HERE / "data_overview.txt")
