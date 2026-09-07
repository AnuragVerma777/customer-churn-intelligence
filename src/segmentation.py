"""KMeans customer segmentation with data-driven labels."""
from __future__ import annotations
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def segment_customers(df: pd.DataFrame, max_clusters: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = df.select_dtypes(include="number").columns.tolist()
    if len(numeric) < 2 or len(df) < 8: raise ValueError("At least two numeric behavior/value columns and eight rows are required for segmentation.")
    X = df[numeric].fillna(df[numeric].median()); scaled = StandardScaler().fit_transform(X)
    choices = range(2, min(max_clusters, len(df) - 1) + 1); best_k = 2; best_score = -1
    for k in choices:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled); score = silhouette_score(scaled, labels)
        if score > best_score: best_k, best_score = k, score
    labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(scaled); out = df.copy(); out["segment_id"] = labels
    profile = out.groupby("segment_id")[numeric].mean().reset_index(); profile["customers"] = out.groupby("segment_id").size().values; profile["segment_label"] = [f"Segment {i}" for i in profile.segment_id]
    return out, profile
