from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # repo root
OUTPUTS = ROOT / "outputs"
ARTIFACTS = OUTPUTS / "artifacts"
PLOTS = OUTPUTS / "plots"
CALIB = OUTPUTS / "calibration"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)
CALIB.mkdir(parents=True, exist_ok=True)

BEST_CKPT = ARTIFACTS / "best_cnn_lct.pt"
SCALE_FILE = ARTIFACTS / "scaler_stats.npz"
CALIB_FILE = CALIB / "calibration_val.npz"
METRICS_JSON = OUTPUTS / "metrics.json"
