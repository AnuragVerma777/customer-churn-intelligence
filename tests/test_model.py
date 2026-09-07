import numpy as np
import pandas as pd
from src.churn_model import train_models
from src.revenue_analysis import revenue_at_risk

def test_model_and_revenue_risk():
    rng = np.random.default_rng(2); n = 40
    df = pd.DataFrame({"value": rng.normal(50, 5, n), "usage": rng.normal(10, 2, n), "target": [0, 1] * 20})
    bundle = train_models(df, "target"); assert bundle["best_name"]
    scores = pd.DataFrame({"customer_value": [100, 200], "churn_probability": [.1, .8], "risk_category": ["LOW", "CRITICAL"]})
    assert revenue_at_risk(scores)["revenue_at_risk"] == 170
