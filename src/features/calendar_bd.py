# src/features/calendar_bd.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

# External deps:
#   pip install holidays
#   pip install hijridate
import holidays
from hijridate import Gregorian   # keep this to match your notebook

BD_WEEKEND = {4, 5}  # Fri=4, Sat=5

def _holiday_maps(dt: pd.Series) -> Dict[str, Dict]:
    years = sorted(dt.dt.year.unique())
    bd_h = holidays.country_holidays("BD", years=years)

    udates = dt.dt.date.unique()
    one_day = pd.Timedelta(days=1)

    is_h_map  = {d: int(d in bd_h) for d in udates}
    name_map  = {d: (bd_h.get(d, "") if d in bd_h else "") for d in udates}
    pre_map   = {d: int((pd.Timestamp(d) + one_day).date() in bd_h) for d in udates}
    post_map  = {d: int((pd.Timestamp(d) - one_day).date() in bd_h) for d in udates}

    # Ramadan via Hijri month==9
    def is_ramadan_date(d):
        h = Gregorian(d.year, d.month, d.day).to_hijri()
        return int(h.month == 9)
    ram_map = {d: is_ramadan_date(d) for d in udates}

    # Durga Puja heuristic (by holiday name)
    lower = {d: (name_map[d].lower() if isinstance(name_map[d], str) else "") for d in udates}
    puja_map = {
        d: int(("durga" in lower[d]) or ("vijay" in lower[d]) or ("bijoya" in lower[d]))
        for d in udates
    }

    return dict(is_h=is_h_map, pre=pre_map, post=post_map, ram=ram_map, puja=puja_map)

def add_bd_calendar_features(df: pd.DataFrame, ts_col: str = "timestamp",
                             one_hot_dow: bool = True,
                             drop_raw_time_cols: bool = True) -> pd.DataFrame:
    """
    Add Bangladesh-specific calendar features.
    Produces: dow_* one-hots (if enabled), is_weekend, is_holiday_bd, is_pre_holiday,
              is_post_holiday, is_ramadan, is_durga_puja.
    Flags are int8 to save memory.
    """
    if ts_col not in df.columns:
        raise ValueError(f"'{ts_col}' not in DataFrame.")
    out = df.copy()
    dt = pd.to_datetime(out[ts_col], errors="raise")

    # discrete basics
    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["weekofyear"] = dt.dt.isocalendar().week.astype(int)

    if one_hot_dow:
        dow_oh = pd.get_dummies(out["day_of_week"], prefix="dow", dtype=np.int8)
        out = pd.concat([out, dow_oh], axis=1)

    out["is_weekend"] = out["day_of_week"].isin(BD_WEEKEND).astype(np.int8)

    maps = _holiday_maps(dt)
    out["is_holiday_bd"]  = dt.dt.date.map(maps["is_h"]).astype(np.int8)
    out["is_pre_holiday"] = dt.dt.date.map(maps["pre"]).astype(np.int8)
    out["is_post_holiday"]= dt.dt.date.map(maps["post"]).astype(np.int8)
    out["is_ramadan"]     = dt.dt.date.map(maps["ram"]).astype(np.int8)
    out["is_durga_puja"]  = dt.dt.date.map(maps["puja"]).astype(np.int8)

    if drop_raw_time_cols:
        out = out.drop(columns=["hour","day_of_week","month","weekofyear"])
    return out
