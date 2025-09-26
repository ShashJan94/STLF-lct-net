from __future__ import annotations
import pandas as pd
from meteostat import Point
from src.data_io.read_load_csv import read_load_csv
from src.weather.align_hourly import align_and_fill_weather

def merge_weather(df_load: pd.DataFrame, lat=23.8103, lon=90.4125, tz="Asia/Dhaka") -> pd.DataFrame:
    idx = pd.DatetimeIndex(df_load["timestamp"])
    wx = align_and_fill_weather(idx, Point(lat, lon), tz=tz, model=True)
    merged = df_load.copy()
    merged[["temperature","humidity","wind_speed"]] = wx[["temperature","humidity","wind_speed"]].to_numpy()
    assert len(merged) == len(df_load)
    assert merged["timestamp"].equals(df_load["timestamp"])
    return merged
