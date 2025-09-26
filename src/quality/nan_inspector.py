from __future__ import annotations
import pandas as pd

def locate_load_nans(df: pd.DataFrame, ts_col: str = "timestamp", y_col: str = "load"):
    """Return (nan_positions_df, consecutive_nan_runs_df) and print a short summary."""
    df = df.sort_values(ts_col).reset_index(drop=True).copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="raise")

    isna = df[y_col].isna()
    nan_positions = df.loc[isna, [ts_col, y_col]].copy()
    nan_positions = nan_positions.rename(columns={ts_col: "timestamp", y_col: "load_nan"})
    # keep dtype explicit; values remain NaN
    nan_positions["load_nan"] = nan_positions["load_nan"].astype("float64")

    df2 = df.reset_index().rename(columns={"index": "row"})
    isna2 = df2[y_col].isna()
    grp = isna2.ne(isna2.shift(fill_value=False)).cumsum()

    runs = (
        df2[isna2]
        .groupby(grp[isna2])
        .agg(
            start_row=("row", "min"),
            end_row=("row", "max"),
            start_ts=(ts_col, "min"),
            end_ts=(ts_col, "max"),
            hours=("row", "size"),
        )
        .reset_index(drop=True)
    )

    if not runs.empty:
        prev_idx = (runs["start_row"] - 1).clip(lower=0)
        next_idx = (runs["end_row"] + 1).clip(upper=len(df2) - 1)
        runs["prev_ts"]  = df2.loc[prev_idx, ts_col].to_numpy()
        runs["prev_val"] = df2.loc[prev_idx, y_col].to_numpy()
        runs["next_ts"]  = df2.loc[next_idx, ts_col].to_numpy()
        runs["next_val"] = df2.loc[next_idx, y_col].to_numpy()
        runs["same_day"] = (runs["start_ts"].dt.date == runs["end_ts"].dt.date).astype(int)

    total_nans = int(isna.sum())
    by_runs = int(runs["hours"].sum()) if not runs.empty else 0
    print(f"Total NaNs in '{y_col}': {total_nans} | sum(hours) across runs: {by_runs}")
    return nan_positions, runs
