from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader

from .scalers import fit_train_only_scalers, apply_scalers
from .windows import build_windows, build_windows_with_context, build_masked_windows_core
from .datasets import SlidingWindowDS, WeightedDS
from .masking import mark_blackouts, mark_ramp_spikes_train_only

@dataclass
class WindowConfig:
    T: int = 72
    horizon: int = 1
    batch_size: int = 64
    scale_load_in_X: bool = True

def build_masked_windows(X, Y, T: int, H: int, blackout_mask=None, spike_mask=None, w_spike: float = 0.3):
    X = X if isinstance(X, np.ndarray) else X.cpu().numpy()
    Y = Y if isinstance(Y, np.ndarray) else Y.cpu().numpy()
    N = len(X); M = N - T - H + 1
    if M <= 0:
        raise ValueError(f"Not enough rows for windows: N={N}, T={T}, horizon={H}")
    if blackout_mask is None: blackout_mask = np.zeros(N, dtype=bool)
    if spike_mask   is None: spike_mask   = np.zeros(N, dtype=bool)

    keep_idx, weights = [], []
    for i in range(M):
        tgt = np.arange(i+T, i+T+H)
        if blackout_mask[tgt].any():
            continue
        weights.append(w_spike if spike_mask[tgt].any() else 1.0)
        keep_idx.append(i)

    Xw, Yw, W = build_masked_windows_core(X, Y, T, H, keep_idx, weights)
    return torch.as_tensor(Xw), torch.as_tensor(Yw), torch.as_tensor(W)

def make_loaders_for_fold(
    train_df, val_df, test_df,
    window_cfg: WindowConfig,
    feature_cols: Iterable[str],
    target_col: str = "load",
    use_context_val: bool = True,
    use_context_test: bool = True,
    mask_train: bool = False,
    w_spike: float = 0.3,
    spike_q: float = 0.999,
    scale_load_in_X: bool = True,
):
    # scalers from TRAIN only
    fscaler, yscaler, noncyc = fit_train_only_scalers(train_df, feature_cols, target_col)

    Xtr, Ytr, _ = apply_scalers(train_df, feature_cols, target_col, fscaler, yscaler, noncyc, scale_load_in_X)
    Xva, Yva, _ = apply_scalers(val_df,   feature_cols, target_col, fscaler, yscaler, noncyc, scale_load_in_X)
    Xte, Yte, _ = apply_scalers(test_df,  feature_cols, target_col, fscaler, yscaler, noncyc, scale_load_in_X)

    T, H, B = window_cfg.T, window_cfg.horizon, window_cfg.batch_size

    # TRAIN
    if mask_train:
        blk_tr = mark_blackouts(train_df)
        spk_tr, _ = mark_ramp_spikes_train_only(train_df, train_df, q=spike_q)
        Xtr_w, Ytr_w, Wtr = build_masked_windows(Xtr, Ytr, T, H, blk_tr, spk_tr, w_spike=w_spike)
        ds_tr = WeightedDS(Xtr_w, Ytr_w, Wtr)
    else:
        Xtr_w, Ytr_w = build_windows(Xtr, Ytr, T, H)
        ds_tr = SlidingWindowDS(Xtr_w, Ytr_w)

    # VAL
    if use_context_val:
        Xva_w_np, Yva_w_np = build_windows_with_context(Xtr, Ytr, Xva, Yva, T, H)
        ds_va = WeightedDS(Xva_w_np, Yva_w_np, np.ones(len(Yva_w_np), dtype=np.float32))
    else:
        Xva_w, Yva_w = build_windows(Xva, Yva, T, H)
        ds_va = WeightedDS(Xva_w, Yva_w, torch.ones(len(Yva_w), dtype=torch.float32))

    # TEST
    if use_context_test:
        Xctx, Yctx = np.vstack([Xtr, Xva]), np.vstack([Ytr, Yva])
        Xte_w_np, Yte_w_np = build_windows_with_context(Xctx, Yctx, Xte, Yte, T, H)
        ds_te = WeightedDS(Xte_w_np, Yte_w_np, np.ones(len(Yte_w_np), dtype=np.float32))
    else:
        Xte_w, Yte_w = build_windows(Xte, Yte, T, H)
        ds_te = WeightedDS(Xte_w, Yte_w, torch.ones(len(Yte_w), dtype=torch.float32))

    # DataLoaders
    dl_tr = DataLoader(ds_tr, batch_size=B, shuffle=True,  drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=B, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=B, shuffle=False)

    # inverse scaler
    def invert_y(y_scaled):
        y_scaled = y_scaled.detach().cpu().numpy() if torch.is_tensor(y_scaled) else y_scaled
        return yscaler.inverse_transform(y_scaled.reshape(-1,1)).reshape(y_scaled.shape)

    used_feats = [target_col] + [c for c in feature_cols if c != target_col]
    return dl_tr, dl_va, dl_te, invert_y, used_feats
