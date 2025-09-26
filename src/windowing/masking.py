from __future__ import annotations
import numpy as np
import pandas as pd

def mark_blackouts(df: pd.DataFrame, thr: float = 1e-6, min_len: int = 3, col: str = "load"):
    z = (df[col].astype(float).values <= thr).astype(np.int8)
    run_id = np.cumsum(np.r_[1, np.diff(z)] != 0)
    run_len = pd.Series(z).groupby(run_id).transform("size").to_numpy()
    return (z == 1) & (run_len >= min_len)

def mark_ramp_spikes_train_only(train_df: pd.DataFrame, full_df: pd.DataFrame,
                                col: str = "load", q: float = 0.999):
    dtrain = train_df[col].astype(float).diff().abs()
    thr = float(np.nanquantile(dtrain, q))
    spikes = full_df[col].astype(float).diff().abs() > thr
    return spikes.fillna(False).to_numpy(), thr
