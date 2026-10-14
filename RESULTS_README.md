# Results README

This directory records the current result artifacts without replacing historical runs.

## Historical multiclass CNN-LSTM

The original C/M/X CNN-LSTM evaluations are stored under `runs/cnn_lstm/seed_<seed>/`. The consolidated, read-only aggregation is in `results/historical_cnn_lstm/`. These are multiclass results on the original C/M/X-only sequence population and are not strict matched ≥M results.

## Matched population

The verified all-label endpoint manifests and 29-feature table are in `results/matched_experiment/`. They contain the frozen matched population described in `EXPERIMENT_PROTOCOL.md`: 73,278 train, 9,240 validation and 9,165 test endpoints, with 313 ≥M test positives.

## Diagnostics

`results/diagnostics/accelerator_smoke_test.json` records the real-image smoke test. It ran on CPU because the current ROCm environment exposes no device. It is a runtime diagnostic, not a model result.

## Not yet available

No strict matched binary CNN or tabular training results, per-example prediction files, bootstrap intervals, ablation results, calibration figures or final comparison tables have been generated. They must not be reported as completed until run on a usable accelerator and saved with the protocol metadata.
