# src/cli/qc_check.py
from __future__ import annotations
import argparse, json
from pathlib import Path

from src.config.load_config import load_config
from src.data_io.read_load_csv import read_load_csv
from src.features.enrich_weather import merge_weather
from src.quality.nan_inspector import locate_load_nans
from src.quality.qc_report import qc_report
from src.quality.audit import audit_anomalies

def main():
    ap = argparse.ArgumentParser(description="Run QC on merged (load+weather) dataset.")
    ap.add_argument("--config", required=True, help="Path to YAML config (uses ${LOAD_CSV})")
    ap.add_argument("--outdir", default="outputs/qc", help="Directory for QC outputs")
    ap.add_argument("--skip-weather", action="store_true", help="Skip Meteostat merge (expects columns present)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Read load
    df = read_load_csv(cfg["data"]["csv_path"])

    # 2) Merge weather unless skipped
    if not args.skip_weather:
        tz  = cfg["data"].get("timezone", "Asia/Dhaka")
        lat = cfg["data"].get("point_lat", 23.8103)
        lon = cfg["data"].get("point_lon", 90.4125)
        dfm = merge_weather(df, lat=lat, lon=lon, tz=tz)
    else:
        dfm = df

    # 3) NaN positions & runs
    nan_positions, nan_runs = locate_load_nans(dfm, ts_col="timestamp", y_col="load")
    nan_positions.to_csv(outdir / "load_nan_positions.csv", index=False)
    nan_runs.to_csv(outdir / "load_nan_runs.csv", index=False)

    # 4) QC summary (+ anomaly audit)
    qc = qc_report(dfm)
    aud = audit_anomalies(dfm, ts_col="timestamp")
    (outdir / "qc_report.json").write_text(json.dumps(qc, indent=2, default=str))
    (outdir / "audit.json").write_text(json.dumps(aud, indent=2, default=str))

    print(f"QC artifacts written to: {outdir}")
    print(" - load_nan_positions.csv")
    print(" - load_nan_runs.csv")
    print(" - qc_report.json")
    print(" - audit.json")

if __name__ == "__main__":
    main()
