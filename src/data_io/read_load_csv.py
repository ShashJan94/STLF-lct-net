from __future__ import annotations
import pandas as pd

REQUIRED_COLS = {"timestamp", "load"}

def read_load_csv(csv_path: str) -> pd.DataFrame:
    """
    Read load CSV, normalize headers to lowercase snake_case, keep only needed cols,
    parse timestamp/load, sort by time.
    """
    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.astype(str)
          .str.strip()
          .str.replace(r"\s+", "_", regex=True)
          .str.replace(r"[^0-9a-zA-Z_]+", "", regex=True)
          .str.lower()
    )
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]
    if not REQUIRED_COLS.issubset(df.columns):
        missing = REQUIRED_COLS - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df["load"] = pd.to_numeric(df["load"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
