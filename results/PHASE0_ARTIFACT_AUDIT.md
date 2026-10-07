# Phase 0: Artifact Audit

Read-only audit performed before new inference, training, or statistical analysis. Existing prediction and model artifacts were left unchanged.

## Binary model-target coverage

The predefined split manifests contain 759,357 Train, 95,933 Validation, and 94,757 Test rows (header excluded). The feature and label sources each contain 950,047 rows. Existing XGBoost predictions and logistic-regression predictions align in row count and schema across both targets.

| Model | Target | Existing artifact | Rows | Probability | True label | Predicted label | Timestamp | Active-region ID | Sample ID |
|---|---|---|---:|---|---|---|---|---|---|
| XGBoost | ge_c | validation_xgboost.csv; test_xgboost.csv | 95,933 / 94,757 | Yes | Yes | No; thresholded label can be derived | Yes | Yes | Yes (`row_id`) |
| XGBoost | ge_m | validation_xgboost.csv; test_xgboost.csv | 95,933 / 94,757 | Yes | Yes | No; thresholded label can be derived | Yes | Yes | Yes (`row_id`) |
| Logistic Regression | ge_c | validation_logistic_regression.csv; test_logistic_regression.csv | 95,933 / 94,757 | Yes | Yes | No; thresholded label can be derived | Yes | Yes | Yes (`row_id`) |
| Logistic Regression | ge_m | validation_logistic_regression.csv; test_logistic_regression.csv | 95,933 / 94,757 | Yes | Yes | No; thresholded label can be derived | Yes | Yes | Yes (`row_id`) |
| HistGradientBoosting | ge_c | metrics.json; model.joblib; no prediction CSV | Test split: 94,757 | No | Not in a prediction artifact | No | No | No | No |
| HistGradientBoosting | ge_m | metrics.json; model.joblib; no prediction CSV | Test split: 94,757 | No | Not in a prediction artifact | No | No | No | No |
| Extra Trees | ge_c | metrics.json; model.joblib; no prediction CSV | Test split: 94,757 | No | Not in a prediction artifact | No | No | No | No |
| Extra Trees | ge_m | metrics.json; model.joblib; no prediction CSV | Test split: 94,757 | No | Not in a prediction artifact | No | No | No | No |

The four XGBoost/logistic CSVs per target contain `row_id, filename, active_region, timestamp, split, flare, target, probability`. Each model's validation/test files have the expected split row counts. XGBoost also has saved majority-climatology validation/test predictions for both targets. The HistGradientBoosting and Extra Trees metric JSONs contain selected validation thresholds and test ROC-AUC, PR-AUC, TSS, F1, precision, recall, and balanced accuracy, but do not contain per-row predictions or the complete requested metric set. Their fitted models are present.

The checked-in source feature CSV, flare-label file, and split manifests are present. `src/xgb_baseline/data.py` derives active-region IDs and timestamps from filenames, verifies label joins, and rejects active-region overlap across the predefined splits. The split timestamp ranges overlap; these artifacts support grouped evaluation, not a chronological generalization claim.

## CNN-LSTM artifacts

| Artifact | Rows | Probabilities | True labels | Row predictions | Timestamp | Active-region ID |
|---|---:|---|---|---|---|---|
| `cnn_lstm_test_results.json`, `cnn_lstm_classification_report.txt`, `cnn_lstm_confusion_matrix.png` | 2,021 test sequences | No | Aggregate counts/metrics | Aggregate confusion matrix only | No explicit timestamp; frame names can be inspected | `ar` is present in `data/Test_sequences.csv` |

The test sequence file contains 2,021 rows with `ar`, `target`, and ten frame paths. At the time of the initial audit, the saved confusion matrix was `[[1537, 171, 0], [147, 132, 0], [0, 34, 0]]`. Checkpoints (`cnn_lstm_best.pth`, `cnn_lstm_final.pth`, `cnn_lstm.pth`), training history, and configuration are present. The recorded configuration says the existing run already used focal loss (`gamma=0.8`), inverse-square-root class sampling, and class weights; this is not an unweighted baseline.

At the time of the initial audit, the evaluation code had not saved CNN-LSTM per-row probabilities or predictions. The CNN training history is present, but only seven epoch rows were recorded in the checked-in CSV.

## Other artifacts and blockers


## Verified existing binary metrics

The existing per-model JSONs agree with the supplied starting values to the displayed precision for HistGradientBoosting and Extra Trees. Existing XGBoost result JSONs include observation-level and active-region-level summaries; the supplied starting values are present in repository documentation and prediction artifacts exist for direct recomputation. These remain existing reported results until independently recomputed from aligned saved predictions.
The HistGradientBoosting and Extra Trees metric JSONs contain selected validation thresholds and test ROC-AUC, PR-AUC, TSS, F1, precision, recall, and balanced accuracy, but do not contain per-row predictions or the complete requested metric set. Their fitted models are present.
The supplied headline XGBoost values (ge_c ROC-AUC 0.890780; ge_m ROC-AUC 0.939496) are the `active_region_selected` summaries in the XGBoost JSON files. They are not the observation-level values used for the other models. Recalculation from saved observation-level XGBoost probabilities gives ge_c ROC-AUC 0.839249 / PR-AUC 0.660312 / TSS 0.526603 at threshold 0.2, and ge_m ROC-AUC 0.903835 / PR-AUC 0.372434 / TSS 0.580691 at threshold 0.1. The XGBoost JSON's observation-level threshold table and saved probabilities agree. The supplied logistic-regression observation-level ROC-AUC/TSS values are also reproduced. This mixed-unit reporting is a cross-model comparability defect; future comparison tables must use one unit consistently and report active-region metrics separately if retained.

The saved HistGradientBoosting and Extra Trees joblib models emit scikit-learn `InconsistentVersionWarning` when loaded under the fresh environment's scikit-learn 1.9.1 (the artifacts were serialized with 1.9.0). HistGradientBoosting test metrics reproduce exactly. Extra Trees regenerated observation-level ROC-AUC/AP differ from the historical metric JSON by less than 0.000004; the regenerated saved probabilities and their checked metrics are used in the new package, with the version caveat retained.

## GPU inference addendum (2026-10-08)

The `.research-venv` ROCm 7.2 PyTorch build detected an AMD Radeon Graphics device with 16 GB memory. GPU inference of the saved best CNN-LSTM checkpoint completed on all 2,021 test sequences in 30.8 seconds. New probabilities are saved at `experiments/cnn_lstm/multiclass/predictions/test_predictions.csv`; the current confusion matrix is `[[1294, 394, 20], [96, 172, 11], [0, 28, 6]]`, agreeing with the refreshed `cnn_lstm_test_results.json`. CPU and GPU probabilities differed by at most 1.8e-7 on a four-sequence parity check. OVR ROC/PR metrics and 2,000 active-region bootstrap intervals are in `results/cnn_lstm_metrics.json`. These current results supersede the initial audit snapshot above.