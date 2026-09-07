"""Conservative, auditable dataframe cleaning."""
from __future__ import annotations
import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before = {"rows": len(df), "columns": len(df.columns), "missing": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum())}
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.drop_duplicates().reset_index(drop=True)
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        numeric = pd.to_numeric(out[col].str.replace(r"[$,%]", "", regex=True), errors="coerce")
        # Numeric-looking columns are often sparse or contain a few blanks;
        # preserve the numeric interpretation when most non-missing values parse.
        non_missing = out[col].notna().sum()
        if non_missing and numeric.notna().sum() / non_missing >= 0.8: out[col] = numeric
        elif any(k in col.lower() for k in ("date", "time", "signup", "purchase")):
            parsed = pd.to_datetime(out[col], errors="coerce")
            if parsed.notna().mean() >= 0.7: out[col] = parsed
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        if out[col].isna().any(): out[col] = out[col].fillna(out[col].median())
    for col in out.select_dtypes(include=["object", "category"]).columns:
        if out[col].isna().any(): out[col] = out[col].fillna("Unknown")
    after = {"rows": len(out), "columns": len(out.columns), "missing": int(out.isna().sum().sum()), "duplicates": int(out.duplicated().sum())}
    return out, {"before": before, "after": after}
