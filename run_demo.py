#!/usr/bin/env python3
"""Run the portable Lake Erie chlorophyll-a reviewer demo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Suppress informational TensorFlow messages before TensorFlow is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from chl_estimator import estimate_chla, write_summary


PACKAGE_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate chlorophyll-a from an 11-band OLCI NetCDF file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PACKAGE_ROOT / "demo_data" / "erie_olci_demo.nc",
        help="Input NetCDF containing latitude, longitude, and the 11 Rw bands.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "output" / "erie_chla_demo.nc",
        help="Output chlorophyll-a NetCDF path.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PACKAGE_ROOT / "output" / "demo_run_summary.json",
        help="JSON run-summary path.",
    )
    parser.add_argument("--model", type=Path, default=PACKAGE_ROOT / "model" / "OLCI-MERIS_Chla_DNN_final.h5")
    parser.add_argument("--x-scaler", type=Path, default=PACKAGE_ROOT / "model" / "x_scaler_final.joblib")
    parser.add_argument("--y-scaler", type=Path, default=PACKAGE_ROOT / "model" / "y_scaler_final.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = estimate_chla(
        args.input.resolve(),
        args.output.resolve(),
        args.model.resolve(),
        args.x_scaler.resolve(),
        args.y_scaler.resolve(),
    )
    write_summary(summary, args.summary.resolve())
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {args.summary.resolve()}")
    print(
        "Finite pixels: {output_finite_pixels}; mean Chl-a: "
        "{chla_mean_mg_m3:.4f} mg m-3; runtime: {runtime_seconds:.3f} s".format(**summary)
    )


if __name__ == "__main__":
    main()
