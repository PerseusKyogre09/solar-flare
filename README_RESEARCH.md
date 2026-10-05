# Solar-Flare Prediction Research Notes

This document is the paper handoff for generating an IEEE-style research paper from this repository. It records the data, task definitions, models, results, reproducibility commands, and limitations for the experiments. Provide it together with the IEEE LaTeX template.

## 1. Research objective

The project evaluates machine-learning methods for predicting solar-flare activity from solar active-region observations. The current comparison contains an image/sequential neural model and several tabular baselines.

The main scientific comparison is between:

1. CNN-LSTM using image sequences.
2. XGBoost using 29 engineered magnetic features.
3. Logistic regression using the same 29 features.
4. HistGradientBoosting using the same 29 features.
5. Extra Trees using the same 29 features.

These are complementary experiments. The CNN-LSTM is currently a multiclass C/M/X classifier, whereas the tabular models are binary event detectors. Raw accuracy must not be used to claim that one family is definitively superior without reconciling the task definitions.

## 2. Dataset provenance and separation

The primary benchmark is the provided Dryad-derived dataset represented by:

- `data/Lat60_Lon60_Nans0_C1.0_24hr_png_224_features.csv`
- `data/C1.0_24hr_224_png_Labels.txt`
- `data/Train_Data_by_AR_png_224.csv`
- `data/Validation_Data_by_AR_png_224.csv`
- `data/Test_Data_by_AR_png_224.csv`

The feature CSV contains 29 numeric engineered features, legacy label columns, and an image filename. The implemented loader ignores the legacy binary label and joins the filename to the checked-in flare-label file.

The repository also contains:

- `data/sharp_mx-nf-all_2010-01-01_2024-04-30.csv`

This SHARP/NASA-related CSV is **not mixed into the benchmark experiments**. It has a different schema and would require a separate feature extraction, labeling, alignment, and split pipeline. It should be described as a possible future external-validation dataset, not as part of the current training data.

## 3. Label definitions

The flare label is parsed from the letter prefix:

- `ge_c`: positive if the flare is C, M, or X; negative if it is `0`.
- `ge_m`: positive if the flare is M or X; negative if it is `0` or C.

The label thresholds are implemented in `src/xgb_baseline/data.py` and are shared by the tabular experiments.

The CNN-LSTM report uses three multiclass labels: C, M, and X. It is therefore not directly equivalent to either binary tabular task.

## 4. Data splitting and leakage controls

The tabular loader validates and joins the predefined Train, Validation, and Test files. Splits are grouped by active region, and active-region overlap checks are enforced. The split files are not purely chronological: timestamp ranges overlap between splits. The evaluation should therefore be described as active-region-grouped evaluation, not future-time forecasting.

The validation set is used to select the operating threshold by maximum TSS over thresholds 0.1 through 0.9. The selected threshold is then applied once to the untouched test set.

The current dataset summary reports 1,570 active regions, approximately 950,047 observations, and 29 features. The target is highly imbalanced, especially for `ge_m`; therefore ROC-AUC alone is insufficient. Report TSS, PR-AUC, precision, recall, F1, and false-alarm behavior.

## 5. Models

### CNN-LSTM

The CNN-LSTM is the image/sequential model already trained in the repository. Its classification report is in `cnn_lstm_classification_report.txt`. It predicts C, M, and X classes.

Test results:

| Metric | Value |
|---|---:|
| Accuracy | 0.825829 |
| Macro F1 | 0.444940 |
| Weighted F1 | 0.825060 |
| Macro POD/recall | 0.457667 |
| Macro FAR | 0.231867 |
| Macro CSI | 0.367100 |
| Macro TSS | 0.261891 |
| Macro HSS | 0.248024 |

Per-class F1: C = 0.9062, M = 0.4286, X = 0.0000. The X class has only 34 test examples and was not detected by this run; this class-specific weakness must be reported rather than hidden by weighted accuracy.

### XGBoost

XGBoost uses the 29 tabular features with histogram tree training, learning rate 0.05, maximum depth 4, 500 estimators, subsample 0.8, column subsample 0.8, and random seed 42. Native missing-value handling is enabled. The implementation also uses early stopping where supported.

Test results at the validation-selected threshold:

| Target | Threshold | ROC-AUC | PR-AUC | TSS | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ge_c | 0.2 | 0.890780 | 0.852837 | 0.502974 | 0.748538 | 0.621359 | 0.941176 |
| ge_m | 0.1 | 0.939496 | 0.720826 | 0.757143 | 0.500000 | 0.333333 | 1.000000 |

The high recall for `ge_m` is accompanied by a high false-alarm rate and low precision. It should not be presented as perfect prediction.

### Logistic regression

Logistic regression is a simple linear tabular baseline. It uses training-only median imputation and standardization. Its test ROC-AUC/TSS results are:

| Target | ROC-AUC | TSS |
|---|---:|---:|
| ge_c | 0.846086 | 0.537476 |
| ge_m | 0.922494 | 0.581815 |

### HistGradientBoosting

This is a separate scikit-learn histogram gradient-boosting implementation, not XGBoost. It uses 300 maximum iterations, learning rate 0.08, 31 maximum leaf nodes, L2 regularization 1.0, and early stopping.

| Target | Threshold | ROC-AUC | PR-AUC | TSS | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ge_c | 0.2 | 0.833828 | 0.647268 | 0.520046 | 0.589272 | 0.479627 | 0.763903 |
| ge_m | 0.1 | 0.891795 | 0.381298 | 0.537638 | 0.411113 | 0.317494 | 0.583031 |

### Extra Trees

Extra Trees is an independently randomized ensemble of decision trees. It uses 160 trees, square-root feature sampling, minimum leaf size 2, balanced class weights, all available CPU workers, and random seed 42.

| Target | Threshold | ROC-AUC | PR-AUC | TSS | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ge_c | 0.3 | 0.830317 | 0.632153 | 0.505629 | 0.590177 | 0.504228 | 0.711447 |
| ge_m | 0.1 | 0.901546 | 0.313543 | 0.605207 | 0.321461 | 0.208487 | 0.701691 |

## 6. Overall interpretation

XGBoost is the strongest current tabular model by ROC-AUC for both binary targets and by TSS for `ge_m`. Logistic regression is competitive on `ge_c`, while HistGradientBoosting and Extra Trees provide additional model-family comparisons but do not consistently outperform XGBoost.

The CNN-LSTM has strong weighted accuracy mainly because C dominates the multiclass test set. Its macro metrics and zero X-class recall show that class imbalance materially affects performance. The paper should emphasize macro metrics and per-class results.

The five methods should be described as a benchmark comparison, not as five directly interchangeable classifiers, because the CNN-LSTM and tabular models use different input representations and task formulations.

## 7. Overfitting and validity statement

There is no evidence in the saved test metrics of catastrophic failure, and the tabular pipeline prevents active-region overlap between splits. However, the repository does not currently contain a saved train-versus-validation learning-curve analysis for every model. The paper should use cautious wording such as:

> The use of active-region-grouped validation and test partitions reduces leakage risk. Nevertheless, because the predefined timestamp ranges overlap and learning curves were not systematically reported for every model, the results should be interpreted as grouped benchmark performance rather than proof of temporal generalization.

Do not claim that the models are completely free of overfitting. Do state that the test set was held out from threshold selection and final metric reporting.

## 8. Reproducibility commands

Use the repository environment:

```bash
venv/bin/python -m src.train_xgboost --target ge_c
venv/bin/python -m src.train_xgboost --target ge_m
venv/bin/python train_histgb.py --target ge_c
venv/bin/python train_histgb.py --target ge_m
venv/bin/python train_extratrees.py --target ge_c
venv/bin/python train_extratrees.py --target ge_m
```

Existing XGBoost outputs are under `experiments/xgboost/`. HistGradientBoosting outputs are under `experiments/histgb/`. Extra Trees outputs are under `experiments/extratrees/`.

The CNN-LSTM artifacts include `cnn_lstm_final.pth`, `cnn_lstm_best.pth`, `cnn_lstm_classification_report.txt`, training history CSV, plots, and confusion matrices.

The installed PyTorch build is ROCm-enabled (`torch 2.10.0+rocm7.0`), but the current execution environment exposes no GPU device: `torch.cuda.is_available()` returns false and `/dev/kfd` is unavailable. The reported experiments were therefore run on CPU unless otherwise stated.

## 9. Recommended paper structure

Use the following IEEE paper organization:

1. Abstract: objective, dataset, five methods, strongest result, and main limitation.
2. Introduction: motivation for solar-flare prediction and contribution of a multi-model benchmark.
3. Related Work: CNN/LSTM forecasting, magnetic active-region features, tree ensembles, and flare-class prediction.
4. Dataset and Preprocessing: source, 29 features, labels, active-region grouping, imbalance, and NASA SHARP separation.
5. Methods: CNN-LSTM, XGBoost, logistic regression, HistGradientBoosting, and Extra Trees.
6. Experimental Protocol: split policy, validation threshold selection, metrics, random seed, and environment.
7. Results: separate multiclass CNN-LSTM table and binary tabular table; include per-class CNN-LSTM results.
8. Discussion: XGBoost performance, imbalance, false alarms, representation differences, and lack of temporal split.
9. Limitations and Future Work: external SHARP validation, chronological evaluation, calibration, and GPU retraining.
10. Conclusion: concise findings without claiming universal superiority.

## 10. Claims to avoid

- Do not say NASA SHARP data were used for training; they were kept separate.
- Do not compare CNN-LSTM accuracy directly with binary XGBoost accuracy.
- Do not call `ge_m` recall of 1.0 perfect forecasting; precision and false alarms are important.
- Do not claim chronological generalization because timestamp ranges overlap.
- Do not claim that XGBoost and CNN-LSTM use identical input information.
- Do not claim that all overfitting has been ruled out.
