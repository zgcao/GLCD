"""Portable chlorophyll-a inference used by the Nature reviewer demo.

This module preserves the numerical core of
``3-image_est_chl_examples_local.py`` while removing machine-specific paths.
"""

from __future__ import annotations

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import xarray as xr
from tensorflow.keras.models import load_model


WAVELENGTHS = (412, 443, 490, 510, 560, 620, 665, 681, 709, 754, 779)


def _package_versions() -> dict[str, str]:
    names = ("tensorflow", "numpy", "scikit-learn", "joblib", "xarray", "netCDF4", "h5py")
    return {name: version(name) for name in names}


def estimate_chla(
    input_file: Path,
    output_file: Path,
    model_file: Path,
    x_scaler_file: Path,
    y_scaler_file: Path,
) -> dict[str, object]:
    """Estimate chlorophyll-a from an 11-band POLYMER/OLCI NetCDF file."""
    started = time.perf_counter()

    with xr.open_dataset(input_file) as source:
        missing = [f"Rw{wave}" for wave in WAVELENGTHS if f"Rw{wave}" not in source]
        missing += [name for name in ("latitude", "longitude") if name not in source]
        if missing:
            raise ValueError(f"Input is missing required variables: {', '.join(missing)}")

        reflectance = np.stack(
            [source[f"Rw{wave}"].values for wave in WAVELENGTHS], axis=-1
        ).astype(np.float32)
        # The original research script converts water reflectance (Rw) to Rrs.
        rrs = reflectance / np.pi
        image_shape = rrs.shape[:2]
        flat = rrs.reshape(-1, len(WAVELENGTHS))
        valid = np.all(np.isfinite(flat), axis=1)
        if not np.any(valid):
            raise ValueError("Input contains no pixels with all 11 finite reflectance bands")

        x_scaler = joblib.load(x_scaler_file)
        y_scaler = joblib.load(y_scaler_file)
        model = load_model(model_file, compile=False)

        scaled = x_scaler.transform(flat[valid])
        prediction = model(scaled, training=False).numpy()
        values = np.exp(y_scaler.inverse_transform(prediction)).reshape(-1)
        values[(values < 0.01) | (values > 1000.0)] = np.nan

        chla = np.full(flat.shape[0], np.nan, dtype=np.float32)
        chla[valid] = values.astype(np.float32)
        chla = chla.reshape(image_shape)

        spatial_dims = source["latitude"].dims
        result = xr.Dataset(
            data_vars={
                "latitude": (spatial_dims, source["latitude"].values.astype(np.float32)),
                "longitude": (spatial_dims, source["longitude"].values.astype(np.float32)),
                "chla_dnn": (spatial_dims, chla),
            },
            attrs={
                "title": "DNN chlorophyll-a estimate for the Lake Erie reviewer demo",
                "source_input": input_file.name,
                "algorithm": "OLCI-MERIS Chl-a DNN",
                "units_note": "chla_dnn is in mg m-3",
            },
        )
        result["latitude"].attrs.update(units="degrees_north")
        result["longitude"].attrs.update(units="degrees_east")
        result["chla_dnn"].attrs.update(
            long_name="chlorophyll-a concentration", units="mg m-3"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in result.data_vars}
    result.to_netcdf(output_file, engine="netcdf4", encoding=encoding)

    finite = chla[np.isfinite(chla)]
    summary: dict[str, object] = {
        "input_file": input_file.name,
        "output_file": output_file.name,
        "shape": list(image_shape),
        "input_valid_pixels": int(valid.sum()),
        "output_finite_pixels": int(finite.size),
        "chla_min_mg_m3": float(np.min(finite)),
        "chla_mean_mg_m3": float(np.mean(finite)),
        "chla_max_mg_m3": float(np.max(finite)),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    return summary


def write_summary(summary: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
