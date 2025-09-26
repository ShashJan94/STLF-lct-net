from __future__ import annotations
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from src.viz.io import ensure_dir, savefig

def plot_daily_missing_heatmap(df: pd.DataFrame, outdir: str | Path,
                               cols = ("load","temperature","humidity","wind_speed"),
                               fname: str = "daily_missing_heatmap.png") -> str:
    """
    Heatmap of daily missing counts for selected columns. Requires 'timestamp'.
    Only plots columns present in df.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp'")
    have = [c for c in cols if c in df.columns]
    if not have:
        raise ValueError("None of the requested columns found in DataFrame.")

    daily_miss = df.set_index("timestamp")[have].isna().resample("D").sum()

    ensure_dir(outdir)
    plt.figure(figsize=(10, 0.6 + 0.4*len(have)))
    plt.imshow(daily_miss.T.values, aspect="auto", origin="lower")
    plt.yticks(range(len(have)), have)
    plt.title("Daily Missing-Value Counts")
    plt.xlabel("Day"); plt.ylabel("Variable")
    cbar = plt.colorbar(); cbar.set_label("# missing hours")
    return savefig(Path(outdir) / fname)
