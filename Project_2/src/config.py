"""Zentrale Pfade. Alle Skripte starten aus Project_2/ mit:  python -m scripts.run_<name>"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"                # Rohdaten (Parquet) liegen hier
INTERIM = DATA_DIR / "interim"          # Zwischenstaende (grosse Tabellen, nicht in Git)
PROCESSED = DATA_DIR / "processed"      # kleine, fertige Ausgaben (z. B. Flag-Liste)
MODELS = ROOT / "models"                # trainierte Modelle
RESULTS = ROOT / "results"              # Metriktabellen (CSV)
FIGURES = ROOT / "reports" / "figures"  # Plots
