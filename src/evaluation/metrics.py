# src/evaluation/metrics.py
import numpy as np
import torch
from sklearn.metrics import r2_score

def metrics_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "MBE": np.nan,
                "MAPE": np.nan, "R2": np.nan, "nRMSE_range": np.nan, "nRMSE_mean": np.nan}
    diff  = y_pred - y_true
    mse   = float(np.mean(diff**2))
    rmse  = float(np.sqrt(mse))
    mae   = float(np.mean(np.abs(diff)))
    mbe   = float(np.mean(diff))
    eps   = 1e-6
    mape  = float(np.mean(np.abs(diff)/(np.abs(y_true)+eps)))
    try: r2 = float(r2_score(y_true, y_pred))
    except Exception: r2 = float("nan")
    rng = float(np.max(y_true) - np.min(y_true)) if y_true.size else 0.0
    mean_abs = float(np.mean(np.abs(y_true)) + eps)
    nrmse_range = float(rmse / (rng + 1e-6)) if rng > 0 else float("nan")
    nrmse_mean  = float(rmse / mean_abs)
    return {"MSE":mse,"RMSE":rmse,"MAE":mae,"MBE":mbe,"MAPE":mape,"R2":r2,
            "nRMSE_range":nrmse_range,"nRMSE_mean":nrmse_mean}

@torch.no_grad()
def evaluate_loader(loader, invert_y_fn, model, ema=None):
    dev = next(model.parameters()).device
    if ema is not None: ema.apply_to(model)
    model.eval()
    preds, trues = [], []
    for batch in loader:
        xb, yb = batch[:2]
        pred = model(xb.to(dev))
        inv_pred = np.asarray(invert_y_fn(pred), dtype=float).reshape(-1)
        inv_true = np.asarray(invert_y_fn(yb),   dtype=float).reshape(-1)
        preds.append(inv_pred); trues.append(inv_true)
    if ema is not None: ema.restore(model)
    y_pred = np.concatenate(preds) if preds else np.array([], float)
    y_true = np.concatenate(trues) if trues else np.array([], float)
    return metrics_dict(y_true, y_pred), y_true, y_pred
