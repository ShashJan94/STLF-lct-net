from __future__ import annotations
import os, yaml
from pathlib import Path

def expand(s: str) -> str:
    return os.path.expandvars(os.path.expanduser(s))

def load_config(path: str) -> dict:
    cfg = yaml.safe_load(open(path, "r"))
    # expand common paths
    if "data" in cfg and "csv_path" in cfg["data"]:
        cfg["data"]["csv_path"] = expand(cfg["data"]["csv_path"])
    return cfg
