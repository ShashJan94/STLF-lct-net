from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.viz.io import ensure_dir, savefig

_DOW_LABELS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def plot_hour_weekday_heatmap(df: pd.DataFrame, outdir: str | Path, fname: str = "hour_weekday_heatmap.png") -> str:
    """
    Heatmap of mean load by hour (rows) × weekday (cols).
    Expects columns: ['timestamp','load'].
    """
    if "timestamp" not in df or "load" not in df:
        raise ValueError("DataFrame must contain 'timestamp' and 'load' columns.")

    d = df[["timestamp","load"]].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="raise")
    d["dow"] = d["timestamp"].dt.dayofweek     # 0..6
    d["hod"] = d["timestamp"].dt.hour          # 0..23

    mat = d.pivot_table(index="hod", columns="dow", values="load", aggfunc="mean")
    # ensure full grid order
    mat = mat.reindex(index=range(24), columns=range(7))
    Z = mat.values.astype(float)

    ensure_dir(outdir)
    plt.figure(figsize=(7.5, 5.5))
    plt.imshow(Z, aspect="auto", origin="lower")
    plt.xticks(range(7), _DOW_LABELS)
    plt.yticks(range(0,24,2), range(0,24,2))
    plt.title("Mean load by hour × weekday")
    plt.xlabel("Day of week"); plt.ylabel("Hour of day")
    cbar = plt.colorbar()
    cbar.set_label("Load (mean)")
    return savefig(Path(outdir) / fname)
