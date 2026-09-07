"""Revenue-at-risk calculations."""
import pandas as pd

def revenue_at_risk(scores: pd.DataFrame, value_col: str = "customer_value", probability_col: str = "churn_probability") -> dict:
    if value_col not in scores or probability_col not in scores: return {"available": False, "reason": "Revenue/value and churn probability are required."}
    s = scores.copy(); s["weighted_risk"] = s[value_col].fillna(0) * s[probability_col].fillna(0)
    return {"available": True, "total_value": float(s[value_col].sum()), "revenue_at_risk": float(s.weighted_risk.sum()), "high_value_at_risk": float(s.loc[s.risk_category.isin(["HIGH", "CRITICAL"]), "weighted_risk"].sum()), "critical_value_at_risk": float(s.loc[s.risk_category.eq("CRITICAL"), "weighted_risk"].sum())}
