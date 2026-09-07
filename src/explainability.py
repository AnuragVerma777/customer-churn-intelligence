"""Model explanation with feature-importance fallback."""
import pandas as pd

try:
    import shap
except Exception:
    shap = None

def feature_importance(bundle: dict) -> pd.DataFrame:
    model = bundle["best_model"]; estimator = model.named_steps["model"]; pre = model.named_steps["preprocessor"]
    try: names = pre.get_feature_names_out()
    except Exception: names = [f"feature_{i}" for i in range(len(getattr(estimator, "feature_importances_", [])))]
    importance = getattr(estimator, "feature_importances_", None)
    if importance is None: importance = abs(getattr(estimator, "coef_", [[0]])[0])
    return pd.DataFrame({"feature": names[:len(importance)], "importance": importance}).sort_values("importance", ascending=False)

def explain_customer(bundle: dict, X: pd.DataFrame, row_index: int = 0) -> pd.DataFrame:
    """Return signed local contributions when SHAP supports the estimator.

    A feature-importance table is returned when the optional SHAP runtime cannot
    explain the selected pipeline, keeping the dashboard usable in lean installs.
    """
    if shap is None:
        result = feature_importance(bundle).head(10).copy(); result["contribution"] = result["importance"]; return result
    model = bundle["best_model"]; pre = model.named_steps["preprocessor"]; estimator = model.named_steps["model"]
    try:
        transformed = pre.transform(X)
        explainer = shap.Explainer(estimator, transformed)
        values = explainer(transformed[row_index:row_index + 1]).values[0]
        names = pre.get_feature_names_out()
        return pd.DataFrame({"feature": names, "contribution": values}).assign(abs_contribution=lambda d: d.contribution.abs()).sort_values("abs_contribution", ascending=False).head(10).drop(columns="abs_contribution")
    except Exception:
        result = feature_importance(bundle).head(10).copy(); result["contribution"] = result["importance"]; return result
