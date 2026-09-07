"""Streamlit dashboard for Customer Churn & Revenue Intelligence."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

try:
    import streamlit as st
    import plotly.express as px
except ImportError:
    st = None

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.analytics import create_database, grouped_sql, sql_kpis
from src.churn_model import score_customers, save_model, train_models
from src.cleaning import clean_data
from src.explainability import feature_importance
from src.ingestion import choose_raw_csv, load_csv, profile_data
from src.recommendations import recommendations
from src.revenue_analysis import revenue_at_risk
from src.schema_detection import detect_schema, normalize_target
from src.segmentation import segment_customers
from src.utils import write_report

st.set_page_config(page_title="Customer Intelligence", page_icon="📊", layout="wide") if st else None

def main() -> None:
    if st is None: raise RuntimeError("Install dependencies with: pip install -r requirements.txt")
    st.title("Customer Churn & Revenue Intelligence System")
    st.caption("Dataset-adaptive analytics: upload a CSV or place one in data/raw/.")
    uploaded = st.sidebar.file_uploader("Upload customer CSV", type=["csv"])
    raw = uploaded or choose_raw_csv(ROOT / "data" / "raw")
    if raw is None:
        st.info("Upload a CSV or add one to data/raw/. A synthetic sample is included in data/sample/."); return
    try: original = load_csv(raw)
    except ValueError as exc: st.error(str(exc)); return
    cleaned, audit = clean_data(original); detection = detect_schema(cleaned); mapping = detection["mapping"]
    with st.sidebar.expander("Detected schema", expanded=True):
        st.json(mapping)
        for key, value in list(mapping.items()):
            options = ["(none)"] + list(cleaned.columns); selected = st.selectbox(key.replace("_", " ").title(), options, index=(options.index(value) if value in options else 0), key=f"map_{key}")
            mapping[key] = None if selected == "(none)" else selected
    page = st.sidebar.radio("Navigate", ["Overview", "Data Quality", "Customer Analytics", "Churn Prediction", "At-Risk Customers", "Customer Explorer", "SQL Insights", "Downloads"])
    target = mapping.get("churn_target"); target_norm = normalize_target(cleaned[target]) if target else None
    if target and target_norm is not None: cleaned["__churn_target__"] = target_norm.astype(int); model_target = "__churn_target__"
    else: model_target = None
    profile = profile_data(cleaned); conn = create_database(cleaned)
    if page == "Overview":
        kpi = sql_kpis(conn, model_target, mapping.get("revenue")); cols = st.columns(5)
        cols[0].metric("Customers", kpi["total_customers"]); cols[1].metric("Churn rate", f"{kpi.get('churn_rate', 0):.1%}" if model_target else "Unavailable"); cols[2].metric("Revenue", f"{kpi.get('revenue', 0):,.2f}" if mapping.get("revenue") else "Unavailable"); cols[3].metric("Rows", profile["rows"]); cols[4].metric("Duplicates removed", audit["before"]["duplicates"] - audit["after"]["duplicates"])
        st.dataframe(cleaned.head(100), use_container_width=True)
        if not model_target: st.warning("Churn prediction is unavailable because this dataset does not contain a reliable historical churn/attrition target.")
    elif page == "Data Quality":
        st.subheader("Data quality profile"); st.json({"before_cleaning": audit["before"], "after_cleaning": audit["after"], "dtypes": profile["dtypes"]}); st.dataframe(pd.DataFrame({"missing": profile["missing"], "missing_pct": profile["missing_pct"], "unique": profile["unique"]}), use_container_width=True)
        nums = profile["numeric"]
        if nums: st.plotly_chart(px.histogram(cleaned, x=st.selectbox("Numeric column", nums)), use_container_width=True)
    elif page == "Customer Analytics":
        if mapping.get("revenue"): st.plotly_chart(px.histogram(cleaned, x=mapping["revenue"], title="Revenue distribution"), use_container_width=True)
        if model_target and mapping.get("revenue"): st.plotly_chart(px.box(cleaned, x="__churn_target__", y=mapping["revenue"], title="Value by churn outcome"), use_container_width=True)
        try:
            segmented, segment_profile = segment_customers(cleaned); st.subheader("Data-driven segments"); st.dataframe(segment_profile, use_container_width=True)
        except ValueError as exc: st.info(str(exc))
    elif page == "Churn Prediction":
        if not model_target: st.warning("Churn prediction is unavailable because no reliable binary target was detected.")
        else:
            try:
                excluded = [c for c in [mapping.get("customer_id"), mapping.get("date"), target] if c]
                bundle = train_models(cleaned, model_target, excluded); bundle.update(target=model_target, excluded=excluded); save_model(bundle); st.session_state["bundle"] = bundle
                st.dataframe(pd.DataFrame(bundle["metrics"]).T, use_container_width=True); st.bar_chart(feature_importance(bundle).head(15).set_index("feature"))
            except ValueError as exc: st.error(str(exc))
    elif page == "At-Risk Customers":
        bundle = st.session_state.get("bundle")
        if not bundle: st.info("Open Churn Prediction first to train a model.")
        else:
            scores = score_customers(bundle, cleaned, mapping.get("customer_id"), mapping.get("revenue")); st.dataframe(scores.sort_values("churn_probability", ascending=False), use_container_width=True); st.download_button("Download risk scores", scores.to_csv(index=False), "risk_scores.csv", "text/csv"); st.metric("Estimated revenue at risk", f"{revenue_at_risk(scores).get('revenue_at_risk', 0):,.2f}" if mapping.get("revenue") else "Unavailable")
    elif page == "Customer Explorer":
        bundle = st.session_state.get("bundle"); id_col = mapping.get("customer_id");
        if bundle:
            scores = score_customers(bundle, cleaned, id_col, mapping.get("revenue")); selected = st.selectbox("Customer", scores.customer_id); row = cleaned.iloc[scores.index[scores.customer_id.eq(selected)][0]]; st.json(row.to_dict()); st.write(recommendations(row, mapping))
        else: st.info("Train a model to explore customer risk.")
    elif page == "SQL Insights":
        st.json(sql_kpis(conn, model_target, mapping.get("revenue")))
        group = mapping.get("segment") or mapping.get("subscription")
        if group: st.dataframe(grouped_sql(conn, group, model_target, mapping.get("revenue")), use_container_width=True)
    elif page == "Downloads":
        st.download_button("Download cleaned dataset", cleaned.to_csv(index=False), "cleaned_customers.csv", "text/csv")
        report = write_report(ROOT / "reports" / "business_report.md", {"Executive Summary": f"{len(cleaned):,} customers analyzed.", "Data Quality": str(audit), "Schema": str(mapping), "Limitations": "Predictions and revenue-at-risk are estimates; unavailable analyses are omitted."})
        st.download_button("Download business report", report.read_bytes(), "business_report.md", "text/markdown")

if __name__ == "__main__": main()
