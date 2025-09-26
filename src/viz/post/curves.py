import os
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_DIR = "outputs/plots"

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def plot_learning_curves(history, out=f"{DEFAULT_DIR}/learning_curves.png"):
    ep = history["epoch"]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ep, history["val_rmse"], label="val rmse")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("val rmse")
    ax2 = ax1.twinx()
    ax2.plot(ep, history["train_mse"], label="train mse", linestyle="--")
    ax2.set_ylabel("train mse")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.title("learning curves")
    plt.tight_layout()
    _ensure_dir(out); plt.savefig(out, dpi=200); plt.close()
    return out
