from pathlib import Path
import json, numpy as np
from src.config.paths import PLOTS, ARTIFACTS, BEST_CKPT, METRICS_JSON
from src.viz.post.curves import plot_learning_curves
from src.viz.post.diagnostics import plot_pred_vs_true, plot_sequence_overlay, plot_residual_hist
from src.evaluation.metrics import evaluate_loader, metrics_dict
from src.evaluation.calibration import calibrate_on_val_and_save
from src.training import train_model
from src.model.cnn_lct_att import CNN_LCT_Att
from src.windowing.loaders import WindowConfig, make_loaders_for_fold
from src.config.features import FEATS



# train_loader, val_loader, test_loader, invert_y, used_feats = make_loaders_for_fold(...)

model = CNN_LCT_Att(in_feats=len(used_feats), max_len=256, baseline_lags=(1,24,48,72))
model, ema, history, total_secs = train_model(
    model, train_loader, val_loader, invert_y_fn=invert_y,
    epochs=28, lr=5e-4, weight_decay=2e-4, huber_beta=0.5,
    ema_decay=0.995, save_path=str(BEST_CKPT),
    patience=6, min_delta=0.25, es_start_epoch=8, clip_grad=1.0, log_every=1,
    debias_lambda=0.0, report_val_bias_corrected=True, report_val_affine_calibrated=True
)

# metrics
train_metrics, ytr, ytrp = evaluate_loader(train_loader, invert_y, model, ema=ema)
val_metrics,   yva, yvap = evaluate_loader(val_loader,   invert_y, model, ema=ema)
test_metrics,  yte, ytep = evaluate_loader(test_loader,  invert_y, model, ema=ema)
with open(METRICS_JSON, "w") as f:
    json.dump({"train":train_metrics, "val":val_metrics, "test":test_metrics}, f, indent=2)

# plots
plot_learning_curves(history, out=str(PLOTS / "learning_curves.png"))
plot_pred_vs_true(yva, yvap, "Validation: Pred vs True", out=str(PLOTS / "val_scatter.png"))
plot_pred_vs_true(yte, ytep, "Test: Pred vs True", out=str(PLOTS / "test_scatter.png"))
plot_residual_hist(yva, yvap, "Validation residuals", out=str(PLOTS / "val_residuals.png"))
plot_residual_hist(yte, ytep, "Test residuals", out=str(PLOTS / "test_residuals.png"))
plot_sequence_overlay(yva, yvap, start=0, length=240,
                      title="Validation overlay (first 240 points)", out=str(PLOTS / "val_overlay.png"))

# optional: fit and save calibration
a,b = calibrate_on_val_and_save(model, val_loader, invert_y_fn=invert_y, ema=ema,
                                checkpoint_path=str(BEST_CKPT))
print("Saved best model & plots to:", ARTIFACTS, PLOTS)
