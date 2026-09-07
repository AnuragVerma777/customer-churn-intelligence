import pandas as pd
from src.cleaning import clean_data

def test_cleaning_strips_numeric_and_duplicates():
    out, audit = clean_data(pd.DataFrame({" value ": [" $10", " $10", None], "name": [" A ", " A ", None]}))
    assert len(out) == 2 and out["value"].dtype.kind in "fi" and audit["before"]["duplicates"] == 1
