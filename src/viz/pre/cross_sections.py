from __future__ import annotations
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from src.viz.io import ensure_dir, savefig

_DOW = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def plot_hour_month_heatmap(df: pd.DataFrame, outdir: str | Path, fname: str = "hour_x_month_heatmap.png") -> str:
    """Mean load by hour × month. Requires ['timestamp','load']."""
    if not {"timestamp","load"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'timestamp','load'")
    d = df.copy()
    dt = pd.to_datetime(d["timestamp"], errors="raise")
    d["hour"] = dt.dt.hour
    d["month"] = dt.dt.month
    mat = d.pivot_table(index="hour", columns="month", values="load", aggfunc="mean")
    mat = mat.reindex(index=range(24), columns=range(1,13))
    Z = mat.values.astype(float)
    ensure_dir(outdir)
    plt.figure(figsize=(7.8,5.2))
    plt.imshow(Z, aspect="auto", origin="lower")
    plt.xticks(range(12), range(1,13))
    plt.yticks(range(0,24,2), range(0,24,2))
    plt.title("Mean Load by Hour × Month")
    plt.xlabel("Month"); plt.ylabel("Hour")
    plt.colorbar(label="Load")
    return savefig(Path(outdir) / fname)

def plot_hour_weekday_heatmap(df: pd.DataFrame, outdir: str | Path, fname: str = "hour_x_weekday_heatmap.png") -> str:
    """Mean load by hour × weekday. Duplicate of earlier but kept here for bundle callers."""
    if not {"timestamp","load"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'timestamp','load'")
    d = df.copy()
    dt = pd.to_datetime(d["timestamp"], errors="raise")
    d["hour"] = dt.dt.hour
    d["dow"]  = dt.dt.dayofweek
    mat = d.pivot_table(index="hour", columns="dow", values="load", aggfunc="mean")
    mat = mat.reindex(index=range(24), columns=range(7))
    Z = mat.values.astype(float)
    ensure_dir(outdir)
    plt.figure(figsize=(7.5,5.2))
    plt.imshow(Z, aspect="auto", origin="lower")
    plt.xticks(range(7), _DOW)
    plt.yticks(range(0,24,2), range(0,24,2))
    plt.title("Mean Load by Hour × Weekday")
    plt.xlabel("Day of week"); plt.ylabel("Hour")
    plt.colorbar(label="Load")
    return savefig(Path(outdir) / fname)

def plot_daily_profile_band(df: pd.DataFrame, outdir: str | Path, fname: str = "daily_profile_band.png") -> str:
    """Median daily profile with 10–90% band. Requires ['timestamp','load']."""
    if not {"timestamp","load"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'timestamp','load'")
    d = df.copy()
    dt = pd.to_datetime(d["timestamp"], errors="raise")
    d["hod"] = dt.dt.hour
    g = d.groupby("hod")["load"]
    med = g.median(); p10 = g.quantile(0.10); p90 = g.quantile(0.90)
    x = med.index.values
    ensure_dir(outdir)
    plt.figure(figsize=(7.2,4.6))
    plt.fill_between(x, p10.values, p90.values, alpha=0.25, label="10–90%")
    plt.plot(x, med.values, label="Median")
    plt.title("Daily Profile (Median with 10–90% band)")
    plt.xlabel("Hour of day"); plt.ylabel("Load")
    plt.legend(); plt.grid(True, alpha=0.3)
    return savefig(Path(outdir) / fname)

def plot_binned_temp_response(df: pd.DataFrame, outdir: str | Path, fname: str = "temp_binned_response.png") -> str:
    """Mean + 10–90% band of load vs temperature (1°C bins). Requires ['temperature','load']."""
    if not {"temperature","load"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'temperature','load'")
    d = df[["temperature","load"]].dropna().copy()
    d["tbin"] = d["temperature"].round().astype(int)
    gb = d.groupby("tbin")["load"]
    t = gb.mean().index.values
    y = gb.mean().values
    q10 = gb.quantile(0.10).values
    q90 = gb.quantile(0.90).values
    ensure_dir(outdir)
    plt.figure(figsize=(7.2,4.6))
    plt.fill_between(t, q10, q90, alpha=0.25, label="10–90%")
    plt.plot(t, y, label="Mean")
    plt.title("Temperature–Load Response (1°C bins)")
    plt.xlabel("Temperature (°C)"); plt.ylabel("Load")
    plt.legend(); plt.grid(True, alpha=0.3)
    return savefig(Path(outdir) / fname)
