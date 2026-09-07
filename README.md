# Customer Churn & Revenue Intelligence System

An end-to-end, dataset-agnostic Streamlit application for profiling customer CSVs, cleaning data, discovering schemas, analyzing revenue and engagement, segmenting customers, predicting churn when a reliable target exists, scoring risk, and estimating revenue at risk.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python generate_sample.py
streamlit run app.py
pytest -q
```

Upload any CSV in the sidebar, or place one in `data/raw/`. The app scores semantic candidates rather than requiring fixed names such as `Churn` or `MonthlyCharges`; mappings can be corrected in the sidebar. Missing targets, values, dates, or identifiers disable only the affected analyses and explain why.

## Architecture

`src/ingestion.py` handles robust CSV loading and profiling; `schema_detection.py` infers semantic columns; `cleaning.py` performs conservative, auditable cleaning; `churn_model.py` builds leakage-aware scikit-learn pipelines; `segmentation.py`, `analytics.py`, `revenue_analysis.py`, `explainability.py`, and `recommendations.py` provide modular analytics. SQLite is used for KPI and grouped SQL queries, and Joblib persists the best model.

## ML methodology

Logistic Regression, Random Forest, and Gradient Boosting are compared using accuracy, precision, recall, F1, ROC-AUC, and PR-AUC. PR-AUC is the selection metric because accuracy can hide poor minority-class recall in imbalanced churn data. IDs and detected date columns are excluded to reduce leakage. SHAP is optional; feature importance is always available as a fallback.

## Supported patterns and limitations

Common aliases for customer IDs, churn/attrition, revenue/value, tenure, plans, dates, usage, support, and segments are recognized using names, types, and value patterns. Synthetic sample data is clearly labeled. Revenue-at-risk is an estimate (`value × predicted probability`), not guaranteed lost revenue; predictions require enough rows, two target classes, and usable features. Future improvements include drift monitoring, calibrated probabilities, richer temporal features, and role-based deployment.

## Screenshots

Add dashboard screenshots here when deploying.
