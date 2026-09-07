"""SQLite-backed analytics."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

def create_database(df: pd.DataFrame, path: str | Path = "outputs/analytics.sqlite") -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    return conn

def sql_kpis(conn: sqlite3.Connection, target: str | None = None, revenue: str | None = None) -> dict:
    result = {"total_customers": int(pd.read_sql_query("SELECT COUNT(*) n FROM customers", conn).iloc[0, 0])}
    if target:
        q = f'SELECT SUM(CASE WHEN "{target}"=1 THEN 1 ELSE 0 END) churned FROM customers'
        churned = int(pd.read_sql_query(q, conn).iloc[0, 0]); result.update(churned=churned, churn_rate=churned / max(result["total_customers"], 1))
    if revenue:
        result.update(revenue=float(pd.read_sql_query(f'SELECT SUM("{revenue}") v FROM customers', conn).iloc[0, 0] or 0))
    return result

def grouped_sql(conn: sqlite3.Connection, column: str, target: str | None = None, revenue: str | None = None) -> pd.DataFrame:
    fields = [f'"{column}" AS group_name', "COUNT(*) AS customers"]
    if target: fields.append(f'AVG("{target}") AS churn_rate')
    if revenue: fields.append(f'AVG("{revenue}") AS average_value')
    return pd.read_sql_query(f'SELECT {", ".join(fields)} FROM customers GROUP BY "{column}" ORDER BY customers DESC', conn)
