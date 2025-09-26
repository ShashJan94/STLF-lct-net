import numpy as np, torch
from .metrics import metrics_dict

@torch.no_grad()
def evaluate_loader(loader, invert_y_fn, model, ema=None):
    dev = next(model.parameters()).device
    if ema is not None: ema.apply_to(model)
    model.eval()
    preds, trues = [], []
    for batch in loader:
        xb, yb = batch[:2]
        xb = xb.to(dev)
        pred = model(xb)               # [B,H]
        inv_pred = invert_y_fn(pred)   # [B,H]
        inv_true = invert_y_fn(yb)     # [B,H]
        preds.append(np.asarray(inv_pred, dtype=float).reshape(-1))
        trues.append(np.asarray(inv_true, dtype=float).reshape(-1))
    if ema is not None: ema.restore(model)
    y_pred = np.concatenate(preds) if preds else np.array([], float)
    y_true = np.concatenate(trues) if trues else np.array([], float)
    return metrics_dict(y_true, y_pred), y_true, y_pred
