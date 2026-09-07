"""Churn modeling, scoring, and persistence."""
from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .feature_engineering import build_preprocessor

try:
    from xgboost import XGBClassifier
except Exception:  # Optional dependency; the remaining estimators still work.
    XGBClassifier = None

def train_models(df: pd.DataFrame, target: str, excluded: list[str] | None = None) -> dict:
    excluded = excluded or []; y = df[target].astype(int); X = df.drop(columns=[target] + [c for c in excluded if c in df.columns])
    if y.nunique() < 2: raise ValueError("Churn target must contain both positive and negative classes.")
    if len(df) < 20: raise ValueError("At least 20 rows are recommended for model training.")
    strat = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)
    estimators = {"Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"), "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"), "Gradient Boosting": GradientBoostingClassifier(random_state=42)}
    if XGBClassifier is not None:
        estimators["XGBoost"] = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=42)
    scores, fitted = {}, {}
    for name, estimator in estimators.items():
        try:
            pipe = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", estimator)])
            pipe.fit(X_train, y_train); proba = pipe.predict_proba(X_test)[:, 1]; pred = (proba >= 0.5).astype(int)
            scores[name] = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0), "roc_auc": roc_auc_score(y_test, proba) if y_test.nunique() == 2 else None, "pr_auc": average_precision_score(y_test, proba), "confusion_matrix": confusion_matrix(y_test, pred).tolist()}; fitted[name] = pipe
        except Exception: continue
    if not fitted: raise ValueError("No model could be trained on the available features.")
    best_name = max(scores, key=lambda n: scores[n]["pr_auc"])
    return {"models": fitted, "metrics": scores, "best_name": best_name, "best_model": fitted[best_name], "feature_columns": X.columns.tolist()}

def score_customers(bundle: dict, df: pd.DataFrame, id_col: str | None = None, revenue_col: str | None = None) -> pd.DataFrame:
    model = bundle["best_model"]; target = bundle.get("target"); X = df.drop(columns=[target] if target and target in df else [], errors="ignore")
    excluded = [c for c in bundle.get("excluded", []) if c in X]
    proba = model.predict_proba(X.drop(columns=excluded, errors="ignore"))[:, 1]
    out = pd.DataFrame({"customer_id": df[id_col].astype(str).values if id_col else df.index.astype(str), "churn_probability": proba})
    out["risk_category"] = pd.cut(out.churn_probability, [-.01, .25, .5, .75, 1.01], labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    if revenue_col and revenue_col in df: out["customer_value"] = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0).values
    return out

def save_model(bundle: dict, path: str | Path = "models/churn_model.joblib") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True); joblib.dump(bundle, path)
