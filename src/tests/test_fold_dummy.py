# scripts/verify_fold.py
import numpy as np, pandas as pd, torch

from src.windowing.scalers import fit_train_only_scalers, apply_scalers
from src.windowing.windows import build_windows_with_context, build_windows

def run_fold_dummy(tr_df, va_df, dl_tr, dl_va, dl_te, invert_y, feature_cols, cfg, used_context_val=True):
    # 0) loaders are non-empty
    assert len(dl_tr) > 0 and len(dl_va) > 0 and len(dl_te) > 0, "Empty loader — check T/horizon/split sizes"

    # 1) grab a validation batch (WeightedDS -> (X,y,w); SlidingWindowDS -> (X,y))
    batch = next(iter(dl_va))
    xb, yb = batch[0], batch[1]
    print("val batch shapes:", tuple(xb.shape), tuple(yb.shape))  # [B, T, F], [B, H]

    # 2) NaN/Inf checks
    def _bad(t): return torch.isnan(t).any().item() or torch.isinf(t).any().item()
    assert not _bad(xb) and not _bad(yb), "NaN/Inf in validation batch"

    # 3) Rebuild the FIRST validation window exactly (respecting context flag)
    fsc, ysc, noncyc = fit_train_only_scalers(tr_df, feature_cols, target_col="load")
    Xtr, Ytr, _ = apply_scalers(tr_df, feature_cols, "load", fsc, ysc, noncyc, scale_load_in_X=cfg.scale_load_in_X)
    Xva, Yva, _ = apply_scalers(va_df, feature_cols, "load", fsc, ysc, noncyc, scale_load_in_X=cfg.scale_load_in_X)

    if used_context_val:
        Xva_w, Yva_w = build_windows_with_context(Xtr, Ytr, Xva, Yva, cfg.T, cfg.horizon)  # numpy
    else:
        Xva_w, Yva_w = build_windows(Xva, Yva, cfg.T, cfg.horizon)  # torch tensors
        Xva_w, Yva_w = Xva_w.numpy(), Yva_w.numpy()

    # Compare the first window/target in loader vs. rebuilt arrays
    tol = 1e-6
    assert np.allclose(xb[0].cpu().numpy(), Xva_w[0].astype(np.float32), atol=tol), "Window X mismatch (val[0])"
    assert np.allclose(yb[0].cpu().numpy(), Yva_w[0].astype(np.float32), atol=tol), "Window y mismatch (val[0])"

    # 4) invert_y round-trip on a slice
    y_inv = invert_y(yb[:8])
    y_fwd = ysc.transform(y_inv.reshape(-1,1)).reshape(y_inv.shape)
    assert np.allclose(y_fwd, yb[:8].cpu().numpy(), atol=1e-6), "invert_y round-trip failed"

    # 5) naive last-value baseline (teacher-forced): X[..., 'load'] last vs y
    load_idx = feature_cols.index("load")
    last_obs_scaled = xb[:, -1, load_idx:load_idx+1]   # [B, 1], scaled like y
    y_hat_inv = invert_y(last_obs_scaled)
    y_true_inv = invert_y(yb)
    mae = float(np.mean(np.abs(y_hat_inv - y_true_inv)))
    p50 = float(np.median(np.abs(y_hat_inv - y_true_inv)))
    print(f"naive last-value baseline — MAE={mae:.3f}, medianAE={p50:.3f} (original scale)")

    # 6) tiny preview
    preview = pd.DataFrame({"y_true": y_true_inv[:10].ravel(), "y_hat_last": y_hat_inv[:10].ravel()})
    print(preview.round(3).to_string(index=False))
    print("✅ Fold dummy verification passed.")
