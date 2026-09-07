import pandas as pd
from src.ingestion import profile_data

def test_profile():
    p = profile_data(pd.DataFrame({"a": [1, 1], "b": [None, 2]}))
    assert p["rows"] == 2 and p["duplicates"] == 0 and p["missing"]["b"] == 1
