import numpy as np, pandas as pd
from . import __all__ as _; _  # no-op to please linters
from src.config.features import FEATS
from src.config.paths import SCALE_FILE

def fit_or_load_minmax(df: pd.DataFrame):
    df = df[FEATS].astype(float)
    if SCALE_FILE.exists():
        s = np.load(SCALE_FILE)
        x_min, x_max = s["x_min"], s["x_max"]
        y_min, y_max = float(s["y_min"]), float(s["y_max"])
    else:
        x_min = df.min(axis=0).to_numpy()
        x_max = df.max(axis=0).to_numpy()
        y_min, y_max = float(df["load"].min()), float(df["load"].max())
        np.savez_compressed(SCALE_FILE, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    return x_min, x_max, y_min, y_max

def prepare_new_data(df: pd.DataFrame, T: int):
    assert all(c in df.columns for c in FEATS), f"df must have {FEATS}"
    if len(df) < T: raise ValueError(f"df has {len(df)} rows; need T={T}")
    x_min, x_max, y_min, y_max = fit_or_load_minmax(df)
    rng = np.maximum(x_max - x_min, 1e-8)
    X_last = df[FEATS].astype(float).iloc[-T:].to_numpy()
    X_scaled = (X_last - x_min) / rng
    import torch
    X = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)  # [1,T,d]
    def invert_y_fn(y_scaled): return (np.asarray(y_scaled)*(y_max - y_min) + y_min)
    return X, invert_y_fn
