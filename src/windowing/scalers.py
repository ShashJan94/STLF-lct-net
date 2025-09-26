from __future__ import annotations
from typing import Iterable, Tuple, Set, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

CYCLIC: Set[str] = {
    "hour_sin","hour_cos","day_sin","day_cos","month_sin","month_cos","woy_sin","woy_cos"
}

def noncyclic_features(feature_cols: Iterable[str]) -> List[str]:
    skip = set(CYCLIC) | {c for c in feature_cols if str(c).startswith(("is_","dow_"))}
    return [c for c in feature_cols if c not in skip and c != "load"]

def fit_train_only_scalers(
    train_df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str = "load",
) -> Tuple[MinMaxScaler, MinMaxScaler, Set[str]]:
    noncyc = set(noncyclic_features(feature_cols))
    fscaler = MinMaxScaler()
    if noncyc:
        fscaler.fit(train_df[list(noncyc)].values)
    else:
        # no-op scaler
        fscaler.min_, fscaler.scale_ = np.zeros((0,)), np.ones((0,))
        fscaler.data_min_ = fscaler.data_max_ = fscaler.data_range_ = np.zeros((0,))
        fscaler.n_features_in_ = 0
    yscaler = MinMaxScaler().fit(train_df[[target_col]].values)
    return fscaler, yscaler, noncyc

def apply_scalers(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    fscaler: MinMaxScaler,
    yscaler: MinMaxScaler,
    noncyc_set: Set[str],
    scale_load_in_X: bool = True,
):
    X = df[list(feature_cols)].copy()
    to_scale = [c for c in feature_cols if c in noncyc_set]
    if to_scale:
        X[to_scale] = fscaler.transform(X[to_scale].values)
    if scale_load_in_X and "load" in X.columns:
        X[["load"]] = yscaler.transform(X[["load"]].values)
    y = yscaler.transform(df[[target_col]].values)
    return X.values.astype(np.float32), y.astype(np.float32), yscaler
