from __future__ import annotations
import torch
from torch.utils.data import Dataset

class SlidingWindowDS(Dataset):
    def __init__(self, Xw, yw):
        self.Xw = Xw if torch.is_tensor(Xw) else torch.as_tensor(Xw, dtype=torch.float32)
        self.yw = yw if torch.is_tensor(yw) else torch.as_tensor(yw, dtype=torch.float32)
    def __len__(self): return self.Xw.shape[0]
    def __getitem__(self, i): return self.Xw[i], self.yw[i]

class WeightedDS(Dataset):
    def __init__(self, X, y, w):
        self.X = X if torch.is_tensor(X) else torch.as_tensor(X, dtype=torch.float32)
        self.y = y if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.float32)
        self.w = w if torch.is_tensor(w) else torch.as_tensor(w, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i], self.w[i]
