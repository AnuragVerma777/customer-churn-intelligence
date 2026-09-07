"""Shared utilities."""
from pathlib import Path
import pandas as pd

def write_report(path: str | Path, sections: dict[str, str]) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Customer Churn & Revenue Intelligence Report\n\n" + "\n\n".join(f"## {k}\n\n{v}" for k, v in sections.items())
    path.write_text(text, encoding="utf-8"); return path
