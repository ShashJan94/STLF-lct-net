from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List

WEATHER_COLS = ("temperature", "humidity", "wind_speed")

def _to_iso_list(ts: pd.Series, n: int = 10) -> List[str]:
    ts = pd.to_datetime(ts, errors="coerce").dropna()
    return [t.isoformat() for t in ts.head(n)]

def audit_anomalies(df: pd.DataFrame, ts_col: str = "timestamp") -> Dict[str, Any]:
    """
    Quick anomaly audit on load and weather. Returns a JSON-safe dict.
    Expects lowercase columns and a datetime-like `ts_col`.
    """
    out: Dict[str, Any] = {}
    d = df.sort_values(ts_col).copy()
    d[ts_col] = pd.to_datetime(d[ts_col], errors="raise")

    # --- Zero / near-zero load (blackouts) ---
    zmask = (d["load"] <= 1e-6)
    out["zero_load_count"] = int(zmask.sum())
    out["zero_load_when"]  = _to_iso_list(d.loc[zmask, ts_col])

    # --- Hour-to-hour ramp spikes (p99.9 on |Δload|) ---
    ramp = d["load"].astype(float).diff().abs()
    thr = float(ramp.quantile(0.999)) if len(ramp) else float("nan")
    spk = ramp > thr if np.isfinite(thr) else pd.Series(False, index=d.index)
    out["ramp_p999"]        = thr
    out["ramp_spike_count"] = int(spk.sum())
    out["ramp_spike_when"]  = _to_iso_list(d.loc[spk, ts_col])

    # --- Weather extremes (0.5–99.5% bounds and max) ---
    for c in WEATHER_COLS:
        if c in d.columns:
            q = d[c].quantile([0.005, 0.01, 0.99, 0.995])
            if np.isfinite(q).all():
                out[f"{c}_Q005_Q995"] = (float(q.iloc[0]), float(q.iloc[-1]))
                out[f"{c}_max"] = float(d[c].max())

    return out
