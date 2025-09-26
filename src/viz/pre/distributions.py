from __future__ import annotations
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from src.viz.io import ensure_dir, savefig

def plot_ramp_histogram(df: pd.DataFrame, outdir: str | Path, fname: str = "ramp_hist.png", bins: int = 100) -> str:
    """Histogram of hourly Δload. Requires ['timestamp','load']."""
    if not {"timestamp","load"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'timestamp','load'")
    L = df.sort_values("timestamp")["load"].astype(float).to_numpy()
    dL = np.diff(L)
    ensure_dir(outdir)
    plt.figure(figsize=(7,4))
    plt.hist(dL, bins=bins)
    plt.title("Hourly ramp distribution (Δload)")
    plt.xlabel("ΔLoad"); plt.ylabel("Count")
    return savefig(Path(outdir) / fname)

def plot_load_duration_curve(df: pd.DataFrame, outdir: str | Path, fname: str = "load_duration_curve.png") -> str:
    """Load duration curve over full span. Requires ['timestamp','load']."""
    if not {"timestamp","load"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'timestamp','load'")
    s = df["load"].astype(float).sort_values().reset_index(drop=True)
    pct = np.linspace(0, 100, len(s), endpoint=True)
    ensure_dir(outdir)
    plt.figure(figsize=(7,4))
    plt.plot(100 - pct, s.values)
    plt.title("Load Duration Curve")
    plt.xlabel("Percent of hours exceeded"); plt.ylabel("Load")
    plt.grid(True, alpha=0.3)
    return savefig(Path(outdir) / fname)


def plot_load_histogram(df: pd.DataFrame, outdir: str | Path,
                        fname: str = "load_hist.png", bins: int = 30,
                        show_kde: bool = False) -> str:
    """
    Histogram of load. Optional very-light KDE using numpy (not seaborn).
    Requires ['timestamp','load'].
    """
    if not {"timestamp","load"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'timestamp','load'")
    x = df["load"].dropna().astype(float).to_numpy()

    ensure_dir(outdir)
    plt.figure(figsize=(8, 5))
    counts, edges, _ = plt.hist(x, bins=bins, alpha=0.8)
    plt.title("Histogram of Load"); plt.xlabel("Load"); plt.ylabel("Frequency")
    if show_kde and len(x) > 50:
        # quick KDE via Gaussian kernel on grid (bandwidth ~ Silverman-lite)
        g = np.linspace(x.min(), x.max(), 512)
        bw = 1.06 * x.std(ddof=1) * (len(x) ** (-1/5)) if x.std() > 0 else (edges[1]-edges[0])
        if bw > 0:
            k = np.exp(-0.5 * ((g[None, :] - x[:, None]) / bw)**2) / (bw * np.sqrt(2*np.pi))
            kde = k.mean(axis=0)
            # scale KDE to histogram area
            area = counts.sum() * (edges[1] - edges[0])
            plt.plot(g, kde * area, linewidth=2)
    return savefig(Path(outdir) / fname)