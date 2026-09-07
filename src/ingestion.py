"""Robust CSV ingestion and profiling helpers."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_csv(source: str | Path | object) -> pd.DataFrame:
    """Load a CSV path or Streamlit UploadedFile with encoding fallbacks."""
    errors = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            if hasattr(source, "seek"): source.seek(0)
            df = pd.read_csv(source, encoding=encoding)
            if df.empty or len(df.columns) == 0: raise ValueError("CSV is empty")
            return df
        except Exception as exc: errors.append(str(exc))
    raise ValueError("Unable to read CSV. Check delimiter, headers, and encoding. " + errors[-1])

def choose_raw_csv(folder: str | Path = "data/raw") -> Path | None:
    paths = sorted(Path(folder).glob("*.csv"))
    return paths[0] if paths else None

def profile_data(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    return {"rows": len(df), "columns": len(df.columns), "duplicates": int(df.duplicated().sum()),
            "missing": missing.to_dict(), "missing_pct": (missing / max(len(df), 1) * 100).round(2).to_dict(),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "unique": df.nunique(dropna=True).to_dict(),
            "numeric": df.select_dtypes(include="number").columns.tolist(),
            "categorical": df.select_dtypes(include=["object", "category", "bool"]).columns.tolist(),
            "dates": [c for c in df.columns if "date" in str(c).lower() or "time" in str(c).lower()]}
