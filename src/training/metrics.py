from sklearn.metrics import r2_score
import numpy as np

def metrics_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return {"MSE": np.nan,"RMSE": np.nan,"MAE": np.nan,"MBE": np.nan,
                "MAPE": np.nan,"R2": np.nan,"nRMSE_range": np.nan,"nRMSE_mean": np.nan}
    diff = y_pred - y_true
    mse  = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(diff)))
    mbe  = float(np.mean(diff))
    eps  = 1e-6
    mape = float(np.mean(np.abs(diff)/ (np.abs(y_true)+eps)))
    try:   r2 = float(r2_score(y_true, y_pred))
    except Exception: r2 = float("nan")
    rng = float(np.max(y_true) - np.min(y_true)) if y_true.size else 0.0
    mean_abs = float(np.mean(np.abs(y_true)) + eps)
    nrmse_range = float(rmse / (rng + 1e-6)) if rng > 0 else float("nan")
    nrmse_mean  = float(rmse / mean_abs)
    return {"MSE": mse,"RMSE": rmse,"MAE": mae,"MBE": mbe,"MAPE": mape,"R2": r2,
            "nRMSE_range": nrmse_range,"nRMSE_mean": nrmse_mean}
