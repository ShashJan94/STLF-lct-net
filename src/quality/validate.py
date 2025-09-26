from __future__ import annotations
import math
from typing import Dict, Iterable, Tuple, List
import numpy as np
import pandas as pd

CYCLIC_COLS = ("hour_sin","hour_cos","day_sin","day_cos","month_sin","month_cos","woy_sin","woy_cos")

def _range_ok(s: pd.Series, lo: float = -1.001, hi: float = 1.001) -> Tuple[bool, float, float]:
    vmin = float(s.min()); vmax = float(s.max())
    return (vmin >= lo) and (vmax <= hi), vmin, vmax

def validate_feature_frame(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    target_col: str = "load",
    required_weather: Iterable[str] = ("temperature","humidity","wind_speed"),
    check_dow_onehot: bool = True,
    tol_range: float = 0.001,
    require_hourly_monotonic: bool = True,
    raise_on_error: bool = True,
) -> Dict[str, object]:
    """
    Validates your feature-enriched dataframe (post-merge, post-feature-engineering).

    Returns a dict summary. If `raise_on_error=True`, raises AssertionError on failure.
    """
    out: Dict[str, object] = {}

    # --- 0) basic schema ---
    assert ts_col in df.columns, f"Missing '{ts_col}'"
    assert target_col in df.columns, f"Missing '{target_col}'"
    out["is_timestamp_dtype"] = pd.api.types.is_datetime64_any_dtype(df[ts_col])
    if raise_on_error:
        assert out["is_timestamp_dtype"], f"'{ts_col}' must be a datetime dtype"

    # --- 1) missingness in critical columns ---
    crit_cols = [target_col, *required_weather, *CYCLIC_COLS]
    have_crit = [c for c in crit_cols if c in df.columns]
    miss_counts = df[have_crit].isna().sum().to_dict()
    out["missing_counts_critical"] = {k:int(v) for k,v in miss_counts.items()}
    if raise_on_error:
        total_miss = sum(int(v) for v in miss_counts.values())
        assert total_miss == 0, f"Found NaNs in critical columns: { {k:int(v) for k,v in miss_counts.items() if int(v)>0} }"

    # --- 2) cyclic ranges in [-1±tol, 1±tol] ---
    range_report: Dict[str, Tuple[float,float,bool]] = {}
    for c in CYCLIC_COLS:
        if c in df.columns:
            ok, vmin, vmax = _range_ok(df[c], lo=-1.0 - tol_range, hi=1.0 + tol_range)
            range_report[c] = (vmin, vmax, ok)
            if raise_on_error:
                assert ok, f"{c} out of range: [{vmin:.6f}, {vmax:.6f}]"
    out["cyclic_ranges"] = {k: {"min":v[0], "max":v[1], "ok":v[2]} for k,v in range_report.items()}

    # --- 3) DOW one-hot exactly one active per row ---
    dow_cols = [c for c in df.columns if c.startswith("dow_")]
    out["dow_onehot_cols"] = dow_cols
    if check_dow_onehot and len(dow_cols) == 7:
        sums = df[dow_cols].sum(axis=1).astype(int)
        dow_ok = bool((sums == 1).all())
        out["dow_onehot_ok"] = dow_ok
        if raise_on_error:
            assert dow_ok, "DOW one-hots should sum to 1 per row"

    # --- 4) binary flags are 0/1 ---
    bin_cols = [c for c in df.columns if c.startswith(("is_","dow_"))]
    non_binary: List[str] = []
    for c in bin_cols:
        if not df[c].dropna().isin([0,1]).all():
            non_binary.append(c)
    out["non_binary_flags"] = non_binary
    if raise_on_error:
        assert not non_binary, f"Non-binary values in: {non_binary}"

    # --- 5) monotonic hourly cadence (optional) ---
    if require_hourly_monotonic:
        dt = pd.to_datetime(df[ts_col]).sort_values()
        diffs = dt.diff().dropna()
        vc = diffs.value_counts()
        out["diff_top3"] = [(str(ix), int(vc.loc[ix])) for ix in vc.index[:3]]
        one_hour = pd.Timedelta(hours=1)
        non_hourly_ratio = float(1.0 - vc.get(one_hour, 0) / len(diffs)) if len(diffs) else 0.0
        out["non_hourly_ratio"] = non_hourly_ratio
        out["duplicates"] = int(df[ts_col].duplicated().sum())
        out["monotonic_increasing"] = bool(dt.is_monotonic_increasing)
        if raise_on_error:
            assert out["duplicates"] == 0, "Duplicate timestamps detected"
            assert out["monotonic_increasing"], "Timestamps must be monotonic increasing"
            assert non_hourly_ratio == 0.0, f"Found non-hourly gaps (ratio={non_hourly_ratio:.4f})"

    return out
