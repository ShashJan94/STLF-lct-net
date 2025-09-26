# src/features/time_encoding.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable

CYCLIC_COLS = (
    "hour_sin","hour_cos",
    "day_sin","day_cos",
    "month_sin","month_cos",
    "woy_sin","woy_cos",
)

def add_cyclic_time(df: pd.DataFrame, ts_col: str = "timestamp",
                    drop_discrete: bool = True,
                    include_woy: bool = True) -> pd.DataFrame:
    """
    Add cyclic time encodings for hour/day/month (+ optional week-of-year).
    Returns a NEW DataFrame.
    """
    if ts_col not in df.columns:
        raise ValueError(f"'{ts_col}' not in DataFrame columns.")
    out = df.copy()
    dt = pd.to_datetime(out[ts_col], errors="raise")

    # raw discrete
    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek  # Mon=0
    out["month"] = dt.dt.month

    # cyclic encodings
    out["hour_sin"]  = np.sin(2*np.pi*out["hour"]/24.0)
    out["hour_cos"]  = np.cos(2*np.pi*out["hour"]/24.0)
    out["day_sin"]   = np.sin(2*np.pi*out["day_of_week"]/7.0)
    out["day_cos"]   = np.cos(2*np.pi*out["day_of_week"]/7.0)
    out["month_sin"] = np.sin(2*np.pi*out["month"]/12.0)
    out["month_cos"] = np.cos(2*np.pi*out["month"]/12.0)

    if include_woy:
        woy = dt.dt.isocalendar().week.astype(int).clip(1, 53)  # safety
        out["woy_sin"] = np.sin(2*np.pi*woy/52.0)
        out["woy_cos"] = np.cos(2*np.pi*woy/52.0)

    if drop_discrete:
        out = out.drop(columns=["hour","day_of_week","month"])
    return out

def has_all_cyclic(df: pd.DataFrame, cols: Iterable[str] = CYCLIC_COLS) -> bool:
    return all(c in df.columns for c in cols)
