# Verification record

Verification date: 24 August 2026 (Asia/Shanghai)

## Clean environment

- Environment location used for testing: a newly created temporary Conda
  prefix, separate from the author's existing research environments
- Operating system: macOS 26.5.2 (build 25F84), arm64
- Python: 3.11.15
- Non-standard hardware: none; CPU execution only
- Fixed direct dependencies: TensorFlow 2.18.0, NumPy 1.26.4,
  scikit-learn 1.3.0, joblib 1.3.2, xarray 2024.10.0, netCDF4 1.7.2,
  and h5py 3.12.1
- Dependency integrity: `python -m pip check` reported no broken requirements
- Clean dependency installation time: 566.21 seconds

## Demo result

Command:

```bash
python run_demo.py
```

Result:

- Exit status: success
- Output dimensions: 64 x 64
- Valid input pixels: 4,096
- Finite output pixels: 4,096
- Minimum Chl-a: 0.38931623 mg m-3
- Mean Chl-a: 0.61135823 mg m-3
- Maximum Chl-a: 1.98489130 mg m-3
- Measured inference-and-output time: 0.067 seconds
- Measured end-to-end command time: 2.09 seconds

The generated files are retained in `output/erie_chla_demo.nc` and
`output/demo_run_summary.json`.

## Integration test

Command:

```bash
python tests/test_demo.py
```

Result: **PASS**. The test completed the full inference in a temporary
directory and verified file creation, variable names, 64 x 64 dimensions,
finite values, and the stated 0.01--1000 mg m-3 validity range. Measured
end-to-end test time was 2.31 seconds.

## Portability audit

No machine-specific absolute paths were found in the portable code or
configuration. Historical Windows drive paths remain only in the unchanged
original script under `source/`, which is retained for transparency and is not
the documented entry point.
