# XGBoost Solar-Flare Baseline

- Target: `ge_m` (`0` is B/no qualifying flare; positive is M, X).
- Features: 29 Dryad engineered feature columns. These are not claimed to be identical to DeFN's 79 features.
- Splits: predefined active-region grouped Train/Validation/Test; overlap checks passed. Timestamp ranges overlap, so this is grouped—not chronological—evaluation.
- Threshold: selected on validation by maximum TSS over `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`; test is evaluated once at that threshold and the configured thresholds.
- Imputation: disabled; XGBoost native missing-value handling is used; logistic regression has its own training-only median imputer.
- Region summary: maximum predicted probability and maximum observed target per active region. This is a region-level operating summary, not an independent-event estimate.

See `dataset_summary.json`, `all_results.json`, prediction CSVs, and feature importance CSV for machine-readable results. Gain and frequency importance are model diagnostics, not causal evidence.

## Limitations

The full experiment completed successfully. Trained models, predictions, metrics, and feature-importance files are stored alongside this report.
