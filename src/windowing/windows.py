from __future__ import annotations
import numpy as np
import torch

def build_windows(X, y, T: int, horizon: int):
    N = len(X); M = N - T - horizon + 1
    if M <= 0:
        raise ValueError(f"Not enough rows for windows: N={N}, T={T}, horizon={horizon}")
    Xw = np.zeros((M, T, X.shape[1]), dtype=np.float32)
    yw = np.zeros((M, horizon), dtype=np.float32)
    for i in range(M):
        Xw[i] = X[i:i+T]
        yw[i] = y[i+T:i+T+horizon, 0]
    return torch.from_numpy(Xw), torch.from_numpy(yw)

def build_windows_with_context(X_prev, y_prev, X_cur, y_cur, T: int, horizon: int):
    Xcat = np.vstack([X_prev[-T:], X_cur]) if len(X_prev) else X_cur
    ycat = np.vstack([y_prev[-T:], y_cur]) if len(y_prev) else y_cur
    N = len(Xcat); M = N - T - horizon + 1
    if M <= 0:
        raise ValueError(f"Not enough rows for windows: N={N}, T={T}, horizon={horizon}")
    Xw = np.zeros((M, T, Xcat.shape[1]), dtype=np.float32)
    yw = np.zeros((M, horizon), dtype=np.float32)
    for i in range(M):
        Xw[i] = Xcat[i:i+T]
        yw[i] = ycat[i+T:i+T+horizon, 0]
    keep = (np.arange(M) + T) < (T + len(y_cur))  # only windows whose target lies in *current* split
    return Xw[keep], yw[keep]

def build_masked_windows_core(X, Y, T: int, H: int, keep_idx, weights):
    K = len(keep_idx)
    Xw = np.zeros((K, T, X.shape[1]), dtype=np.float32)
    Yw = np.zeros((K, H), dtype=np.float32)
    W  = np.ones((K,), dtype=np.float32)
    for j, i in enumerate(keep_idx):
        Xw[j] = X[i:i+T]
        Yw[j] = Y[i+T:i+T+H, 0]
        W[j]  = weights[j]
    return Xw, Yw, W
