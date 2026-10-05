# Solar-flare model comparison

| Model | Input/task | Test result |
|---|---|---|
| CNN-LSTM | Image sequences; multiclass C/M/X | Accuracy 0.826; macro-F1 0.445; macro-TSS 0.262 |
| XGBoost | 29 features; any C/M/X (`ge_c`) | ROC-AUC 0.891; TSS 0.503; recall 0.941 |
| XGBoost | 29 features; any M/X (`ge_m`) | ROC-AUC 0.939; TSS 0.757; recall 1.000 |
| HistGradientBoosting | 29 features; any C/M/X (`ge_c`) | ROC-AUC 0.834; PR-AUC 0.647; TSS 0.520 |
| HistGradientBoosting | 29 features; any M/X (`ge_m`) | ROC-AUC 0.892; PR-AUC 0.381; TSS 0.538 |
| Extra Trees | 29 features; any C/M/X (`ge_c`) | ROC-AUC 0.830; PR-AUC 0.632; TSS 0.506 |
| Extra Trees | 29 features; any M/X (`ge_m`) | ROC-AUC 0.902; PR-AUC 0.314; TSS 0.605 |
| Logistic regression | 29 features; `ge_c` | ROC-AUC 0.846; TSS 0.537 |
| Logistic regression | 29 features; `ge_m` | ROC-AUC 0.922; TSS 0.582 |

The XGBoost runs completed successfully. Their models, predictions, and metrics are under `experiments/xgboost/`.

The HistGradientBoosting runs completed successfully as an additional CPU tabular baseline. Their models and metrics are under `experiments/histgb/`; rerun them with `venv/bin/python train_histgb.py --target ge_c` or `--target ge_m`.

Extra Trees is the fifth method. Its outputs are under `experiments/extratrees/`.

The CNN-LSTM is multiclass, while XGBoost and logistic regression are binary event-detection models. These numbers are therefore complementary experiments, not a strict accuracy ranking. Report the task definitions with every metric.

The provided/Dryad-derived dataset and the NASA SHARP CSV remain separate. XGBoost uses predefined active-region grouped Train/Validation/Test splits, with thresholds selected on validation data and applied once to the test set.
