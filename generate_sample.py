"""Generate a reproducible synthetic dataset for local demonstration."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed); tenure = rng.integers(1, 73, n); usage = np.maximum(1, rng.normal(45, 18, n)); support = rng.poisson(1.3, n); revenue = np.maximum(20, rng.normal(85, 25, n)); logits = -1.7 - tenure * .018 - usage * .012 + support * .3 + revenue * .006; probability = 1 / (1 + np.exp(-logits)); churn = rng.random(n) < probability
    return pd.DataFrame({"Customer ID": [f"C{i:05d}" for i in range(1, n + 1)], "Age": rng.integers(18, 75, n), "Region": rng.choice(["North", "South", "East", "West"], n), "Subscription": rng.choice(["Basic", "Plus", "Premium"], n, p=[.45, .35, .2]), "Tenure": tenure, "Monthly Revenue": revenue.round(2), "Total Revenue": (revenue * tenure).round(2), "Usage": usage.round(1), "Sessions": rng.poisson(12, n), "Support Tickets": support, "Payment Method": rng.choice(["Card", "Bank", "Wallet"], n), "Churn": np.where(churn, "Yes", "No")})

if __name__ == "__main__":
    path = Path(__file__).parent / "data" / "sample" / "synthetic_customers.csv"; path.parent.mkdir(parents=True, exist_ok=True); generate().to_csv(path, index=False); print(path)
