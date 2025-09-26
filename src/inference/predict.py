import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from src.inference.scaling import prepare_new_data
from src.config.paths import CALIB_FILE

class _XOnlyDS(Dataset):
    def __init__(self, X): self.X = X
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i]

@torch.no_grad()
def predict_from_df(model, checkpoint_path, df, T):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    X, invert_y_fn = prepare_new_data(df, T)

    if getattr(model, "use_mha", False) and hasattr(model, "mha") and hasattr(model.mha, "max_len"):
        max_len = int(model.mha.max_len)
        if T > max_len: raise ValueError(f"T={T} exceeds attention max_len={max_len}")

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)

    dl = DataLoader(_XOnlyDS(X), batch_size=1, shuffle=False)
    y_scaled = []
    for xb in dl:
        y = model(xb.to(device)).squeeze(-1).cpu().numpy()
        y_scaled.append(y)
    y_scaled = np.concatenate(y_scaled).reshape(-1)
    y = invert_y_fn(y_scaled.reshape(-1,1)).reshape(-1)

    if CALIB_FILE.exists():
        s = np.load(CALIB_FILE)
        a, b = float(s["a"]), float(s["b"])
        y = a * y + b
    return float(y[0])
