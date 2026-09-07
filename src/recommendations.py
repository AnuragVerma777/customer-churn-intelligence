"""Evidence-based rule recommendations."""
import pandas as pd

def recommendations(row: pd.Series, schema: dict) -> list[str]:
    recs = []; p = float(row.get("churn_probability", 0))
    if p >= .75: recs.append("Prioritize retention outreach based on high predicted risk.")
    usage = schema.get("usage"); support = schema.get("support"); revenue = schema.get("revenue")
    if usage and usage in row and p >= .5: recs.append(f"Review low engagement ({usage}={row[usage]}) and consider an engagement campaign.")
    if support and support in row and p >= .5 and float(row[support]) > 0: recs.append(f"Review support issues ({support}={row[support]}).")
    if revenue and revenue in row and p >= .5: recs.append("Prioritize this higher-value account for a personalized review.")
    return recs or ["Continue monitoring; no strong evidence-based action was identified."]
