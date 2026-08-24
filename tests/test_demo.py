#!/usr/bin/env python3
"""Small end-to-end integration test; no pytest installation is required."""

from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="lake_chla_demo_") as temporary:
        output = Path(temporary) / "result.nc"
        summary = Path(temporary) / "summary.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "run_demo.py"),
                "--output",
                str(output),
                "--summary",
                str(summary),
            ],
            check=True,
            cwd=ROOT,
        )
        assert output.is_file() and output.stat().st_size > 0
        assert summary.is_file() and summary.stat().st_size > 0
        with xr.open_dataset(output) as ds:
            assert set(("latitude", "longitude", "chla_dnn")) <= set(ds.data_vars)
            assert ds["chla_dnn"].shape == (64, 64)
            values = ds["chla_dnn"].values
            assert np.isfinite(values).sum() > 0
            assert np.nanmin(values) >= 0.01
            assert np.nanmax(values) <= 1000.0
    print("PASS: portable Lake Erie demo completed and output checks succeeded")


if __name__ == "__main__":
    main()
