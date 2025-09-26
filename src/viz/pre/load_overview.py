from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.viz.io import ensure_dir, savefig

def plot_weekly_mean_load(df: pd.DataFrame, outdir: str | Path, fname: str = "weekly_mean_load.png") -> str:
    """
    Weekly mean of 'load' over time. Expects columns: ['timestamp','load'].
    Saves PNG and returns its path.
    """
    if "timestamp" not in df or "load" not in df:
        raise ValueError("DataFrame must contain 'timestamp' and 'load' columns.")

    dfi = df[["timestamp","load"]].copy()
    dfi["timestamp"] = pd.to_datetime(dfi["timestamp"], errors="raise")
    dfi = dfi.sort_values("timestamp")
    s = dfi.set_index("timestamp")["load"].resample("W").mean()

    ensure_dir(outdir)
    plt.figure(figsize=(10, 4))
    s.plot()
    plt.title("Weekly mean load")
    plt.xlabel("Date"); plt.ylabel("Load (original units)")
    return savefig(Path(outdir) / fname)
