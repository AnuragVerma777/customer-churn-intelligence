"""Dataset-agnostic semantic column detection."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


ALIASES = {
    "customer_id": ["customer id", "customerid", "client id", "clientid", "user id", "userid", "account id", "accountid", "member id"],
    "churn_target": ["churn", "churned", "is churned", "attrition", "left", "cancelled", "canceled", "retained"],
    "revenue": ["revenue", "sales", "total revenue", "amount", "spend", "charges", "monthly revenue", "customer value", "ltv", "clv"],
    "tenure": ["tenure", "months active", "customer age", "months", "lifetime months", "relationship length"],
    "subscription": ["subscription", "plan", "package", "membership", "tier", "contract"],
    "date": ["date", "signup date", "join date", "created at", "last purchase", "transaction date", "order date", "timestamp"],
    "usage": ["usage", "sessions", "logins", "transactions", "orders", "activity", "engagement", "minutes"],
    "support": ["tickets", "complaints", "support calls", "support tickets", "issues", "contacts"],
    "segment": ["segment", "region", "geography", "country", "state", "city", "customer type"],
}

@dataclass
class Candidate:
    column: str
    score: float
    reason: str

def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

def _target_score(series: pd.Series) -> float:
    vals = {str(v).strip().lower() for v in series.dropna().unique()}
    if 2 <= len(vals) <= 3 and vals & {"yes", "no", "true", "false", "1", "0", "y", "n", "churned", "active", "left", "stayed"}:
        return 0.35
    return 0.0

def detect_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Return high-confidence semantic mappings and ranked candidates."""
    mapping: dict[str, str | None] = {key: None for key in ALIASES}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for semantic, aliases in ALIASES.items():
        ranked: list[Candidate] = []
        for col in df.columns:
            name = _norm(col)
            score = 0.0
            reason = ""
            if name in aliases:
                score += 0.8; reason = "exact alias"
            elif any(alias in name or name in alias for alias in aliases):
                score += 0.5; reason = "name pattern"
            s = df[col]
            if semantic == "customer_id" and (s.nunique(dropna=True) / max(len(s), 1) > 0.85):
                score += 0.15; reason += ", high cardinality"
            if semantic == "churn_target": score += _target_score(s)
            if semantic in {"date"} and pd.api.types.is_datetime64_any_dtype(s): score += 0.3
            if semantic in {"revenue", "tenure", "usage", "support"} and pd.api.types.is_numeric_dtype(s): score += 0.1
            if score > 0: ranked.append(Candidate(str(col), min(score, 1.0), reason.strip(", ")))
        ranked.sort(key=lambda x: x.score, reverse=True)
        candidates[semantic] = [c.__dict__ for c in ranked[:5]]
        if ranked and ranked[0].score >= 0.65:
            mapping[semantic] = ranked[0].column
    return {"mapping": mapping, "candidates": candidates}

def normalize_target(series: pd.Series) -> pd.Series | None:
    """Normalize common binary churn labels to 0/1; return None if ambiguous."""
    text = series.astype(str).str.strip().str.lower()
    positive = {"yes", "y", "true", "1", "churn", "churned", "left", "cancelled", "canceled", "attrited"}
    negative = {"no", "n", "false", "0", "active", "stayed", "retained", "not churned"}
    values = set(text[series.notna()].unique())
    if not values or not (values <= positive | negative) or not (values & positive) or not (values & negative):
        return None
    return text.map(lambda x: 1 if x in positive else 0).astype("Int64")
