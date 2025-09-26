from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

def _month_start(ts) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    return ts.to_period("M").to_timestamp()

@dataclass
class FoldSpec:
    train_start: pd.Timestamp
    train_end_excl: pd.Timestamp
    val_start: pd.Timestamp
    val_end_excl: pd.Timestamp
    test_start: pd.Timestamp

def rolling_quarter_folds(
    df_with_ts: pd.DataFrame,
    ts_col: str = "timestamp",
    start_train_months: int = 24,
    val_months: int = 3,
    step_months: int = 3,
    holdout_test_months: int = 6,
    min_train_days: int = 90,
    min_val_days: int = 21,
    needed_window_hours: int = 0,   # e.g. T + horizon
) -> Tuple[List[Tuple[pd.DataFrame, pd.DataFrame, FoldSpec]], pd.DataFrame]:
    df = df_with_ts.sort_values(ts_col).reset_index(drop=True).copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="raise")

    t0 = pd.to_datetime(df[ts_col].iloc[0])
    tN = pd.to_datetime(df[ts_col].iloc[-1])
    t0_m = _month_start(t0)
    tN_m = _month_start(tN)

    test_start = tN_m - pd.DateOffset(months=holdout_test_months)
    df_trainval = df[df[ts_col] < test_start].copy()
    df_test     = df[df[ts_col] >= test_start].copy()

    first_val_start = t0_m + pd.DateOffset(months=start_train_months)

    folds: List[Tuple[pd.DataFrame, pd.DataFrame, FoldSpec]] = []
    s = first_val_start
    while True:
        e = s + pd.DateOffset(months=val_months)
        if e > test_start:
            break
        if df_trainval.empty or s <= df_trainval[ts_col].iloc[0]:
            s = s + pd.DateOffset(months=step_months)
            continue

        train_df = df_trainval[df_trainval[ts_col] < s].copy()
        val_df   = df_trainval[(df_trainval[ts_col] >= s) & (df_trainval[ts_col] < e)].copy()
        if train_df.empty or val_df.empty:
            s = s + pd.DateOffset(months=step_months)
            continue

        train_days = (train_df[ts_col].iloc[-1] - train_df[ts_col].iloc[0]).days
        val_days   = (val_df[ts_col].iloc[-1]   - val_df[ts_col].iloc[0]).days
        ok_lengths = (train_days >= min_train_days) and (val_days >= min_val_days)

        ok_windows = True
        if needed_window_hours:
            ok_windows = len(train_df) >= needed_window_hours

        if ok_lengths and ok_windows:
            spec = FoldSpec(
                train_start=train_df[ts_col].iloc[0],
                train_end_excl=s,
                val_start=s,
                val_end_excl=e,
                test_start=test_start,
            )
            folds.append((train_df, val_df, spec))

        s = s + pd.DateOffset(months=step_months)

    return folds, df_test

def summarize_folds(folds, ts_col: str = "timestamp") -> str:
    lines = []
    for i, (tr, va, spec) in enumerate(folds, 1):
        lines.append(
            f"#{i:02d} train [{spec.train_start:%Y-%m-%d} … "
            f"{(spec.train_end_excl - pd.Timedelta(seconds=1)):%Y-%m-%d}]"
            f" | val [{spec.val_start:%Y-%m-%d} … {(spec.val_end_excl - pd.Timedelta(seconds=1)):%Y-%m-%d}]"
            f" | rows train={len(tr):,}, val={len(va):,}"
        )
    return "\n".join(lines)
