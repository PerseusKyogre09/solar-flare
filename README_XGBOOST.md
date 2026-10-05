# Scientific XGBoost baseline

This baseline consumes the Dryad `Lat60_Lon60_Nans0_C1.0_24hr_png_224_features.csv` format: 29 numeric engineered magnetic-complexity values, followed by legacy label fields and an image basename. The legacy binary label is ignored. The checked-in `C1.0_24hr_224_png_Labels.txt` is joined by basename and converted explicitly: ≥C = C/M/X, ≥M = M/X, and `0` is negative. The 29 columns are not asserted to be identical to the original DeFN 79 features.

The repository contains the configured 29-feature CSV. The checked-in SHARP CSV/cache is not a substitute: it has a different schema and is intentionally rejected by this pipeline.

Run unit tests first:

```bash
venv/bin/python -m pytest -q tests/test_xgb_baseline.py
```

Install a CPU-compatible XGBoost build in the active environment, then run a small aligned smoke test:

```bash
venv/bin/python -m pip install xgboost
venv/bin/python -m src.train_xgboost --target ge_c --limit-per-split 200
venv/bin/python -m src.train_xgboost --target ge_m --limit-per-split 200
```

The full `ge_c` and `ge_m` experiments have been run. Outputs are under `experiments/xgboost/<target>/`. Threshold selection uses validation TSS only. Observation metrics and an active-region maximum-probability summary are both reported; the latter is not an independent flare-event estimate.
