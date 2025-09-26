# ===== Preflight (model-safe, no edits to your model) =====
import numpy as np
import torch

def run_preflight(model, train_loader, val_loader, invert_y_fn, T_expected=None):
    # 0) Ensure global F points to torch.nn.functional (if someone shadowed it)
    try:
        _ok = hasattr(globals().get("F", None), "softmax")
    except Exception:
        _ok = False
    if not _ok:
        import torch.nn.functional as _F
        globals()["F"] = _F   # restore ONLY the alias; model code remains unchanged

    # 1) grab real batches
    batch_tr = next(iter(train_loader))
    xb_tr, yb_tr = batch_tr[:2]
    wb_tr = batch_tr[2] if len(batch_tr) >= 3 else None

    batch_va = next(iter(val_loader))
    xb_va, yb_va = batch_va[:2]

    B, T, n_feats = xb_va.shape
    H = yb_va.shape[1]
    if T_expected is not None and T != T_expected:
        raise ValueError(f"Window length mismatch: loader T={T}, expected {T_expected}")

    # 2) forward pass + basic checks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    with torch.no_grad():
        yhat = model(xb_va.to(device))
    assert yhat.shape == (B, H), f"Bad output shape {tuple(yhat.shape)} != {(B,H)}"
    def _bad(t): return torch.isnan(t).any().item() or torch.isinf(t).any().item()
    assert not _bad(yhat), "NaN/Inf in model output"
    print(f"forward ok: X[B,T,F]={tuple(xb_va.shape)} → yhat[B,H]={tuple(yhat.shape)}")

    # 3) one train step with weights if present
    model.train(True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-4)
    xb, yb = xb_tr.to(device), yb_tr.to(device)
    wb = wb_tr.to(device) if wb_tr is not None else None

    def weighted_huber(pred, target, weight=None, beta=0.5):
        err = pred - target
        abs_err = err.abs()
        hub = torch.where(abs_err <= beta, 0.5*err**2, beta*(abs_err - 0.5*beta))
        hub = hub.mean(dim=1)
        if weight is not None: hub = hub * weight
        return hub.mean()

    opt.zero_grad(set_to_none=True)
    pred = model(xb)
    loss = weighted_huber(pred, yb, wb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print(f"train-step ok; loss={float(loss):.4f}")

    # 4) quick one-batch overfit (should drop >30%)
    model.train(True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    init, last = None, None
    for i in range(40):
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        l = weighted_huber(pred, yb, wb)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        v = float(l.detach().cpu())
        if init is None: init = v
        last = v
    drop = (init - last) / max(1e-6, init)
    print(f"one-batch overfit: start={init:.4f} end={last:.4f} drop={100*drop:.1f}%")
    assert drop > 0.3, "Overfit didn’t drop >30% — check scaling/learning rate."

    # 5) MAE preview on a val batch (original units)
    model.eval()
    with torch.no_grad():
        pred_val = model(xb_va.to(device))
    yhat_inv = invert_y_fn(pred_val)
    ytrue_inv = invert_y_fn(yb_va.to(device))
    mae_preview = float(torch.mean(torch.abs(torch.as_tensor(yhat_inv) - torch.as_tensor(ytrue_inv))))
    print(f"preview val MAE (orig scale): {mae_preview:.3f}")
    print("✅ preflight passed.")
