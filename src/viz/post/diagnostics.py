import os
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_DIR = "outputs/plots"

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def plot_pred_vs_true(y_true, y_pred, title="pred vs true",
                      out=f"{DEFAULT_DIR}/val_scatter.png"):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=6, alpha=0.6)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("true"); plt.ylabel("predicted"); plt.title(title)
    plt.tight_layout(); _ensure_dir(out); plt.savefig(out, dpi=200); plt.close()
    return out

def plot_residual_hist(y_true, y_pred, title="residuals",
                       out=f"{DEFAULT_DIR}/val_residuals.png"):
    res = np.asarray(y_pred) - np.asarray(y_true)
    plt.figure(figsize=(6, 3.6))
    plt.hist(res, bins=60, alpha=0.8)
    plt.title(title); plt.xlabel("residual (pred - true)"); plt.ylabel("count")
    plt.tight_layout(); _ensure_dir(out); plt.savefig(out, dpi=200); plt.close()
    return out

def plot_sequence_overlay(y_true_seq, y_pred_seq, start=0, length=240,
                          title="overlay",
                          out=f"{DEFAULT_DIR}/val_overlay.png"):
    y_true_seq = np.asarray(y_true_seq); y_pred_seq = np.asarray(y_pred_seq)
    end = min(start + length, len(y_true_seq))
    idx = np.arange(start, end)
    plt.figure(figsize=(9, 3.5))
    plt.plot(idx, y_true_seq[start:end], label="true")
    plt.plot(idx, y_pred_seq[start:end], label="pred", alpha=0.9)
    plt.legend(); plt.title(title); plt.tight_layout()
    _ensure_dir(out); plt.savefig(out, dpi=200); plt.close()
    return out
