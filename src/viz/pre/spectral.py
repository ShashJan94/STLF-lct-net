from __future__ import annotations
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from src.viz.io import ensure_dir, savefig

def _require(modname: str):
    try:
        return __import__(modname, fromlist=['*'])
    except Exception as e:
        raise RuntimeError(f"Required package '{modname}' not found. Install it to use this plot.") from e

def plot_acf_load(df: pd.DataFrame, outdir: str | Path,
                  fname: str = "acf_load.png", lags: int = 336) -> str:
    """Autocorrelation of load up to `lags` hours. Requires ['timestamp','load']."""
    if not {"timestamp","load"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'timestamp','load'")
    sm = _require("statsmodels.tsa.stattools")
    x = df["load"].astype(float).to_numpy()
    x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)

    acf_vals = sm.acf(x, nlags=lags, fft=True, missing="conservative")
    idx = np.arange(acf_vals.size)
    n_eff = np.isfinite(x).sum()
    conf = 1.96 / np.sqrt(max(n_eff, 1))

    ensure_dir(outdir)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.vlines(idx, 0.0, acf_vals, linewidth=1)
    ax.plot(idx, acf_vals, '.', markersize=2)
    ax.set_title(f"Autocorrelation of Load (up to {lags} lags)")
    ax.set_xlabel("Lag (hours)"); ax.set_ylabel("ACF")
    ax.grid(True, alpha=0.3)
    ax.axhspan(-conf, conf, color='gray', alpha=0.15, linewidth=0)
    for k in [24, 168, 336]:
        if k <= lags:
            ax.axvline(k, color='k', linestyle=':', linewidth=0.8)
    plt.tight_layout()
    return savefig(Path(outdir) / fname)

def plot_periodogram_load(df: pd.DataFrame, outdir: str | Path,
                          fname: str = "periodogram_load.png") -> str:
    """Periodogram (power vs period in hours). Requires ['timestamp','load']."""
    if not {"timestamp","load"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'timestamp','load'")
    sp = _require("scipy.signal")
    x = df["load"].astype(float).to_numpy()
    fs = 1.0  # samples/hour
    f, Pxx = sp.periodogram(x, fs=fs, detrend="linear", scaling="density")
    mask = f > 0

    ensure_dir(outdir)
    plt.figure(figsize=(8, 4.5))
    plt.semilogy(1.0/f[mask], Pxx[mask])
    plt.title("Periodogram (Power vs Period)")
    plt.xlabel("Period (hours)"); plt.ylabel("Power Spectral Density")
    plt.grid(True, which="both", alpha=0.3)
    return savefig(Path(outdir) / fname)
