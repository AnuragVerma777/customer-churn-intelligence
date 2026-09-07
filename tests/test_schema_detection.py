import pandas as pd
from src.schema_detection import detect_schema, normalize_target

def test_alias_detection_and_target():
    df = pd.DataFrame({"User ID": [1, 2], "Attrition": ["Left", "Stayed"], "Sales": [10, 20]})
    m = detect_schema(df)["mapping"]
    assert m["customer_id"] == "User ID" and m["churn_target"] == "Attrition" and m["revenue"] == "Sales"
    assert normalize_target(df["Attrition"]).tolist() == [1, 0]
