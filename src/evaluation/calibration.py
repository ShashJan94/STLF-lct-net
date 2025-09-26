# src/evaluation/calibration.py
import os, numpy as np, torch
from .metrics import metrics_dict, evaluate_loader

DEFAULT_CALIB_DIR = "outputs/calibration"
os.makedirs(DEFAULT_CALIB_DIR, exist_ok=True)

def apply_affine(y_pred, a, b):
    p = np.asarray(y_pred, dtype=float).ravel()
    return a * p + b

def load_calibration(path):
    z = np.load(path)
    return float(z["a"]), float(z["b"])

@torch.no_grad()
def calibrate_on_val_and_save(model, val_loader, invert_y_fn, *,
                              ema=None, checkpoint_path=None,
                              calib_path=f"{DEFAULT_CALIB_DIR}/calibration_val.npz"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=True)

    _, y_val, p_val = evaluate_loader(val_loader, invert_y_fn, model, ema=ema)
    y = np.asarray(y_val).ravel()
    p = np.asarray(p_val).ravel()
    a, b = np.polyfit(p, y, 1)
    os.makedirs(os.path.dirname(calib_path), exist_ok=True)
    np.savez_compressed(calib_path, a=a, b=b)
    print(f"[calibration] saved {calib_path}: y_cal = {a:.6f} * y_pred + {b:.6f}")
    return float(a), float(b)

def evaluate_with_affine(model, loader, invert_y_fn, a, b, *, ema=None):
    raw_metrics, y, p = evaluate_loader(loader, invert_y_fn, model, ema=ema)
    p_cal = apply_affine(p, a, b)
    cal_metrics = metrics_dict(y, p_cal)
    return raw_metrics, cal_metrics, y, p, p_cal
