from __future__ import annotations
import numpy as np
import pandas as pd

def qc_report(df: pd.DataFrame,
              ts_col: str = "timestamp",
              load_col: str = "load",
              weather_cols = ("temperature","humidity","wind_speed")) -> dict:
    out = {}
    df = df.sort_values(ts_col).copy()
    dt = pd.to_datetime(df[ts_col], errors="raise")
    out["timespan"] = (dt.min(), dt.max(), int(len(df)))

    deltas = dt.diff().dropna().value_counts(normalize=True)
    out["freq_top3"] = {k: float(v) for k, v in deltas.head(3).items()}
    out["non_hourly_ratio"] = float(1 - deltas.get(pd.Timedelta(hours=1), 0.0))

    out["duplicates"] = int(df[ts_col].duplicated().sum())
    out["monotonic_increasing"] = bool(dt.is_monotonic_increasing)

    cols_present = [c for c in (load_col, *weather_cols) if c in df.columns]
    miss = df[cols_present].isna().mean().to_dict()
    out["missing_frac"] = {k: float(v) for k, v in miss.items()}

    L = df[load_col].astype(float).to_numpy()
    out["load_min_max_mean"] = (float(np.nanmin(L)), float(np.nanmax(L)), float(np.nanmean(L)))
    out["load_zero_ratio"]   = float(np.mean(L == 0))
    out["load_neg_count"]    = int(np.sum(L < 0))

    dL = np.diff(L)
    if len(dL):
        q = np.quantile(np.abs(dL), [0.95, 0.99, 0.995, 0.999])
        out["ramp_abs_quantiles"] = {k: float(v) for k, v in zip(["p95","p99","p99.5","p99.9"], q)}
        out["ramp_mean_std"] = (float(np.mean(dL)), float(np.std(dL)))
        out["ramp_extreme_idxs"] = int(np.sum(np.abs(dL) > q[-1]))
    else:
        out["ramp_abs_quantiles"] = {}
        out["ramp_mean_std"] = (0.0, 0.0)
        out["ramp_extreme_idxs"] = 0

    if "temperature" in df.columns:
        ok = df[[load_col,"temperature"]].dropna()
        out["corr_load_temp_pearson"] = float(ok[load_col].corr(ok["temperature"])) if len(ok) > 100 else float("nan")

    for c in weather_cols:
        if c in df.columns:
            ok = df[c].dropna()
            if len(ok):
                out[f"{c}_min_max_mean"] = (float(ok.min()), float(ok.max()), float(ok.mean()))
    return out
