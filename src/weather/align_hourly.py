from __future__ import annotations
import numpy as np
import pandas as pd
from meteostat import Point, Hourly

KEEP_MAP = {"temp": "temperature", "rhum": "humidity", "wspd": "wind_speed"}

def fetch_point_hourly(point: Point, t_start: pd.Timestamp, t_end: pd.Timestamp,
                       tz: str = "Asia/Dhaka", model: bool = True) -> pd.DataFrame:
    """
    Fetch hourly weather for a Point, convert to local tz (naive), round to hour, dedup.
    """
    wx = Hourly(point, t_start, t_end, model=model).fetch()
    if wx is None or wx.empty:
        raise RuntimeError("Meteostat Point query returned no data.")

    if wx.index.tz is None:
        wx.index = wx.index.tz_localize("UTC")
    wx.index = wx.index.tz_convert(tz).tz_localize(None)

    wx = wx.copy()
    wx.index = wx.index.round("h")                 # hour rounding
    wx = wx[~wx.index.duplicated(keep="first")]    # drop dups

    present = [k for k in KEEP_MAP if k in wx.columns]
    wx = wx[present].rename(columns=KEEP_MAP).copy()
    for c in ["temperature", "humidity", "wind_speed"]:
        if c not in wx.columns:
            wx[c] = np.nan
    return wx

def align_and_fill_weather(load_index: pd.DatetimeIndex,
                           point: Point,
                           tz: str = "Asia/Dhaka",
                           model: bool = True) -> pd.DataFrame:
    """
    Reindex weather to exactly the load timestamps and fill NaNs by hour-of-day then global median.
    """
    start = load_index.min()
    end   = load_index.max() + pd.Timedelta(hours=1)
    wx = fetch_point_hourly(point, start, end, tz=tz, model=model)

    # align exactly
    wx = wx.reindex(load_index)

    # fill NaNs: hour-of-day medians → global medians
    hod = pd.Series(load_index.hour, index=load_index)
    for c in ["temperature", "humidity", "wind_speed"]:
        by_hod = wx.groupby(hod)[c].transform(lambda s: s.fillna(s.median()))
        wx[c] = wx[c].fillna(by_hod)
        if wx[c].isna().any():
            wx[c] = wx[c].fillna(wx[c].median())
    return wx
