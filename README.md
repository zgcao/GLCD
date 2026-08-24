# Global Lake Chlorophyll-a: Nature code-review demo

This repository is a compact, reviewer-ready demonstration of the
chlorophyll-a (Chl-a) inference step used in the associated global lake study.
It contains source code, trained inference assets, and a small real Lake Erie
dataset. The demo reads 11 Sentinel-3 OLCI/POLYMER water-reflectance bands and
writes a NetCDF file containing DNN-estimated Chl-a.

The portable implementation follows the numerical workflow in the original
research script `source/3-image_est_chl_examples_local.py`: convert `Rw` to
remote-sensing reflectance by dividing by pi, apply the fitted input scaler,
run the trained Keras model, inverse-transform and exponentiate its output, and
retain estimates from 0.01 to 1000 mg m-3. The original script is included
unchanged for transparency. It contains historical Windows drive paths and is
not the recommended entry point; `run_demo.py` is the portable wrapper.

## Repository contents

```text
.
|-- README.md
|-- LICENSE
|-- DATA_AND_MODEL.md
|-- requirements.txt
|-- environment.yml
|-- chl_estimator.py
|-- run_demo.py
|-- demo_data/
|   `-- erie_olci_demo.nc
|-- model/
|   |-- OLCI-MERIS_Chla_DNN_final.h5
|   |-- x_scaler_final.joblib
|   `-- y_scaler_final.joblib
|-- source/
|   `-- 3-image_est_chl_examples_local.py
|-- tests/
|   `-- test_demo.py
`-- output/
```

## System requirements

- Operating system: macOS, Linux, or Windows supported by Python and TensorFlow
- Python: 3.11
- Memory: less than 1 GB for the included 64 x 64 demo
- Disk space: approximately 1 GB temporarily during environment installation;
  the repository itself is much smaller
- Non-standard hardware: none. A CPU is sufficient; no GPU, cluster, or other
  specialized hardware is required.

The package was tested on **macOS 26.5.2, Apple Silicon (arm64)** with Python
3.11 in a clean isolated environment. Exact tested package versions are pinned
in `requirements.txt` and recorded by each run in
`output/demo_run_summary.json`.

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate global-lake-chla-demo
```

### Python virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The verified clean installation took **566 seconds (9 minutes 26 seconds)** on
the tested connection; allow approximately 10 minutes. TensorFlow is the
largest dependency, so the time is mainly download-dependent. Installation does
not compile the project source code.

## Run the demo

From the repository root:

```bash
python run_demo.py
```

No paths need to be edited. The command uses the included Lake Erie subset and
writes:

- `output/erie_chla_demo.nc`: `latitude`, `longitude`, and `chla_dnn`
- `output/demo_run_summary.json`: dimensions, numerical summary, runtime,
  platform, Python version, and dependency versions

Expected end-to-end runtime on the tested system is approximately **2.1
seconds**. The measured inference-and-output portion was 0.067 seconds; the
remainder was mainly Python/TensorFlow startup. The first TensorFlow import may
take longer than subsequent runs.

Expected output dimensions are 64 x 64 pixels. A successful run prints the
number of finite estimates, mean Chl-a, and elapsed time. Exact values from the
verified run are reported in `VALIDATION.md`.

## Run the integration test

```bash
python tests/test_demo.py
```

The test runs the full inference workflow in a temporary directory and checks
that the output exists, has the expected variables and 64 x 64 dimensions,
contains finite Chl-a values, and respects the stated 0.01--1000 mg m-3 range.

## Run on user data

Provide a NetCDF file with 2-D `latitude` and `longitude` variables and these
11 identically shaped 2-D water-reflectance variables:

`Rw412`, `Rw443`, `Rw490`, `Rw510`, `Rw560`, `Rw620`, `Rw665`, `Rw681`,
`Rw709`, `Rw754`, and `Rw779`.

Then run:

```bash
python run_demo.py --input /path/to/input.nc --output /path/to/chla_output.nc \
  --summary /path/to/run_summary.json
```

The input `Rw` variables are expected to be dimensionless, fully normalized
water reflectance, matching the POLYMER-derived inputs used to train and apply
the model. Pixels lacking any required band are written as `NaN`. Large scenes
may require more memory because the 11 bands are loaded into memory for
inference.

Optional `--model`, `--x-scaler`, and `--y-scaler` arguments can point to other
compatible assets. Run `python run_demo.py --help` for all options.

## Code and data availability

- **Source code:** portable inference and test code are included at repository
  root and under `tests/`; the unchanged original research script is under
  `source/`.
- **Demo dataset:** `demo_data/erie_olci_demo.nc` is a small real Sentinel-3A
  Lake Erie subset containing only the variables required for inference.
- **Model and preprocessing:** the Keras model and both fitted scalers are under
  `model/`.
- **License:** repository software is released under the MIT License. See
  `DATA_AND_MODEL.md` for data and model provenance and data-license scope.
- **Integrity:** SHA-256 hashes of the demo data, model assets, and unchanged
  original script are listed in `CHECKSUMS.sha256`.

## Notes for reproducibility

The fitted scaler files are Python serialized objects and should only be loaded
from this trusted repository. They were created with scikit-learn 1.3.0, which
is pinned to avoid cross-version serialization warnings. Numerical results can
vary slightly across processor architectures and TensorFlow builds, but output
dimensions, validity checks, and scientific units should remain unchanged.
