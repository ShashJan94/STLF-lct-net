from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def savefig(path: str | Path, dpi: int = 160) -> str:
    path = str(path)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path
