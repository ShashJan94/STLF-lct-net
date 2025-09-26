from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.viz.io import ensure_dir, savefig

try:
    from scipy.ndimage import gaussian_filter  # optional
except Exception:
    gaussian_filter = None

def plot_temp_load_magma_dense(
    df: pd.DataFrame,
    outdir: str | Path,
    fname: str = "temp_load_dense.png",
    gridsize: int = 300,
    sigma: float = 1.3,
    jitter_x: float = 0.10,
    jitter_y: float = 2.0,
    clip_percent: float = 99.5,
    sqrt_contrast: bool = True,
    cmap: str = "turbo"
) -> str:
    """
    Smoothed 2D density of Load vs Temperature with small jitter and optional Gaussian blur.
    Saves the figure and returns the path.
    Requires columns: ['timestamp','load','temperature'].
    """
    d = df.sort_values("timestamp").copy()
    if not {"timestamp","load","temperature"}.issubset(d.columns):
        raise ValueError("DataFrame must contain 'timestamp','load','temperature'")

    # cosmetic filtering: remove outages + top 0.1% ramps
    m = (d["load"] > 0)
    ramp = d["load"].astype(float).diff().abs()
    thr = float(np.nanquantile(ramp, 0.999)) if len(ramp) else np.inf
    m &= ramp.fillna(0).le(thr)

    t = d.loc[m, "temperature"].to_numpy(dtype=float)
    y = d.loc[m, "load"].to_numpy(dtype=float)

    t1, t2 = float(np.nanmin(t)), float(np.nanmax(t))
    y1, y2 = float(np.nanmin(y)), float(np.nanmax(y))

    # jitter to break quantization banding
    rng = np.random.default_rng(42)
    t = t + rng.normal(0, jitter_x, size=t.shape)
    y = y + rng.normal(0, jitter_y, size=y.shape)

    xbins = np.linspace(t1, t2, gridsize + 1)
    ybins = np.linspace(y1, y2, gridsize + 1)
    H, _, _ = np.histogram2d(t, y, bins=[xbins, ybins])

    # smooth if SciPy available
    Hs = H
    if gaussian_filter is not None and sigma and sigma > 0:
        Hs = gaussian_filter(H, sigma=sigma, mode="nearest")

    if sqrt_contrast:
        Hs = np.sqrt(Hs)

    vmax = np.percentile(Hs, clip_percent)
    vmin = 0.0

    ensure_dir(outdir)
    plt.figure(figsize=(8.2, 5.8))
    im = plt.imshow(
        Hs.T, origin="lower", extent=(t1, t2, y1, y2),
        aspect="auto", cmap=cmap, interpolation="bilinear",
        vmin=vmin, vmax=vmax
    )
    cbar = plt.colorbar(im); cbar.set_label("smoothed density" + (" (sqrt)" if sqrt_contrast else ""))
    plt.xlabel("Temperature (°C)"); plt.ylabel("Load (original units)")
    plt.title("Load vs Temperature (smoothed density)")
    return savefig(Path(outdir) / fname)
