import torch

def print_train_masking_stats(train_loader, T, H, train_rows_for_fold):
    ds_tr = train_loader.dataset
    has_w = hasattr(ds_tr, "w")
    if not has_w:
        print("Train dataset has no weights attribute -> masking is OFF for train.")
        return

    w = ds_tr.w if torch.is_tensor(ds_tr.w) else torch.as_tensor(ds_tr.w)
    print(
        "train windows:", len(ds_tr),
        "| weight stats -> min:", float(w.min()), "max:", float(w.max()),
        "mean:", float(w.mean()),
        "spike-weighted frac:", float((w < 0.999).float().mean())
    )
    # crude pre-mask window count: M0 ≈ N - T - H + 1
    M0 = max(0, train_rows_for_fold - T - H + 1)
    drop_pct = 100.0 * (1.0 - len(ds_tr) / max(1, M0))
    print("approx. drop % due to blackout removal:", drop_pct)
