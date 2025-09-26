import math, time, copy, numpy as np, torch, torch.nn.functional as F
from .ema import EMA
from .eval import evaluate_loader
from .metrics import metrics_dict

def train_model(model, train_loader, val_loader, invert_y_fn,
                epochs=60, lr=5e-4, weight_decay=2e-4, huber_beta=0.5,
                ema_decay=0.995, save_path="best_model.pt",
                patience=5, min_delta=0.25, es_start_epoch=None,
                clip_grad=1.0, log_every=1,
                debias_lambda=0.0,
                report_val_bias_corrected=True,
                report_val_affine_calibrated=True):

    device = next(model.parameters()).device
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    ema = EMA(model, decay=ema_decay)

    best_state, best_val_rmse, es_wait = None, float("inf"), 0
    prev_lr = opt.param_groups[0]["lr"]

    # expose defaults for downstream inference
    if not hasattr(model, "val_offset"):   model.val_offset = 0.0
    if not hasattr(model, "val_affine_a"): model.val_affine_a = 1.0
    if not hasattr(model, "val_affine_b"): model.val_affine_b = 0.0

    history = {"epoch": [], "secs": [], "lr": [],
               "train_loss": [], "train_mse": [],
               "val_rmse": [], "val_mae": [],
               "val_mbe": [], "val_mape": [], "val_r2": []}

    t0 = time.time()
    for ep in range(1, epochs+1):
        ep_t0 = time.time()
        model.train()
        train_losses = []
        for batch in train_loader:
            xb, yb = batch[:2]
            wb = batch[2] if len(batch) >= 3 else None
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            wb = wb.to(device) if wb is not None else None

            opt.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(-1)      # [B]
            err = pred - yb[:, 0]
            abs_err = err.abs()
            huber = torch.where(abs_err <= huber_beta, 0.5*err**2,
                                huber_beta*(abs_err - 0.5*huber_beta))
            if wb is not None: huber = huber * wb
            loss = huber.mean()
            if debias_lambda > 0: loss = loss + debias_lambda * err.mean().pow(2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            ema.update(model)
            train_losses.append(float(loss.detach().cpu()))

        # validation
        val_metrics, yv, pv = evaluate_loader(val_loader, invert_y_fn, model, ema=ema)
        if ep % max(1, int(log_every)) == 0:
            raw = val_metrics
            print(
                f"\nEpoch {ep:03d} | train_loss={np.mean(train_losses):.6f} | "
                f"VAL RAW  RMSE={raw['RMSE']:.3f} MAE={raw['MAE']:.3f} "
                f"MSE={raw['MSE']:.3f} MBE={raw['MBE']:.3f} MAPE={raw['MAPE']:.4f} R2={raw['R2']:.4f} "
                f"nRMSErng={raw['nRMSE_range']:.4f} nRMSEmean={raw['nRMSE_mean']:.4f}"
            )
            if yv.size and pv.size and report_val_bias_corrected:
                b_star = float(np.mean(pv - yv))
                pv_corr = pv - b_star
                corr = metrics_dict(yv, pv_corr)
                print(f"           VAL +OFF RMSE={corr['RMSE']:.3f} MAE={corr['MAE']:.3f} "
                      f"MBE={corr['MBE']:.3f}  (offset={b_star:+.3f})")
                model.val_offset = b_star
            if yv.size and pv.size and report_val_affine_calibrated:
                a, b = np.polyfit(pv, yv, 1)
                pv_aff = a*pv + b
                aff = metrics_dict(yv, pv_aff)
                print(f"           VAL +AFF RMSE={aff['RMSE']:.3f} MAE={aff['MAE']:.3f} "
                      f"MBE={aff['MBE']:.3f}  (y≈{a:.6f}*pred+{b:.6f})")

            print(f"           LR={opt.param_groups[0]['lr']:.2e} | epoch_secs={time.time()-ep_t0:.1f}s")

        # scheduler + ES on RAW RMSE
        sched.step(val_metrics["RMSE"])
        new_lr = opt.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"(LR) reduced: {prev_lr:.2e} → {new_lr:.2e}; ES streak reset")
            es_wait = 0
        prev_lr = new_lr

        gate = (es_start_epoch is None) or (ep >= es_start_epoch)
        improved = (val_metrics["RMSE"] + (min_delta if gate else 0.0)) < best_val_rmse
        if improved:
            best_val_rmse = val_metrics["RMSE"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)
            es_wait = 0
        else:
            if gate:
                es_wait += 1
                if es_wait >= patience:
                    print(f"Early stopping at epoch {ep}: best RAW RMSE={best_val_rmse:.3f}")
                    break

        # book-keeping
        history["epoch"].append(ep)
        history["secs"].append(time.time()-ep_t0)
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        history["train_loss"].append(avg_train)
        history["train_mse"].append(avg_train)
        history["val_rmse"].append(float(val_metrics["RMSE"]))
        history["val_mae"].append(float(val_metrics["MAE"]))
        history["val_mbe"].append(float(val_metrics["MBE"]))
        history["val_mape"].append(float(val_metrics["MAPE"]))
        history["val_r2"].append(float(val_metrics["R2"]))

    total_secs = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n=== Training finished ===")
    print(f"Best RAW Val RMSE: {best_val_rmse:.3f}")
    if history["val_rmse"]:
        i = -1
        print(f"Last epoch RAW Val: RMSE={history['val_rmse'][i]:.3f}, "
              f"MAE={history['val_mae'][i]:.3f}, MBE={history['val_mbe'][i]:.3f}, "
              f"MAPE={history['val_mape'][i]:.4f}, R2={history['val_r2'][i]:.4f}")
    print(f"Total training time: {total_secs:.1f}s")

    return model, ema, history, total_secs
