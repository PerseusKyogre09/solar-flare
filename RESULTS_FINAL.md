# Final Results: Solar-Flare Benchmark

This report summarizes verified outputs produced from the repository's saved artifacts. It separates observation-level point estimates from active-region-cluster bootstrap uncertainty. The NASA/SHARP CSV was not used for training or evaluation.

## Dataset and protocol

The feature file and flare-label file each contain 950,047 rows and 29 generic numeric feature columns. The predefined grouped manifests contain 759,357 Train, 95,933 Validation, and 94,757 Test rows. Test predictions cover 157 active regions. The split loader checks for active-region overlap, but timestamp ranges overlap, so the evaluation is grouped benchmark performance, not chronological forecasting.

For each tabular model/target, the threshold is selected on Validation from 0.1 to 0.9 by maximum TSS, with recall as the tie-break, then frozen for Test. Model comparisons use the same 94,757 test rows and identical labels/active-region IDs. Point estimates below are observation-level. The 95% intervals resample 157 active regions with replacement, seed 42, for 2,000 replicates; all rows from a drawn region travel together.

## CNN-LSTM multiclass results

The 2,021-sequence test distribution is C=1708, M=279, X=34. Macro-F1 is 0.466, macro recall 0.517, macro precision 0.461, weighted F1 0.763, macro TSS 0.329, macro HSS 0.242, macro CSI 0.352, macro FAR 0.539, and multiclass MCC 0.310. Accuracy is 0.728 and is secondary because C dominates the sample.

| True class | Precision | Recall | F1 | ROC-AUC OVR | PR-AUC OVR | Support |
|---|---:|---:|---:|---:|---:|---:|
| C | 0.931 | 0.758 | 0.835 | 0.759 | 0.925 | 1708 |
| M | 0.290 | 0.616 | 0.394 | 0.703 | 0.273 | 279 |
| X | 0.162 | 0.176 | 0.169 | 0.875 | 0.085 | 34 |

The held-out confusion matrix records C-to-M=394, M-to-C=96, X-to-C=0, and X-to-M=28; it identified 6 of the 34 X examples. Macro one-vs-rest ROC-AUC is 0.779 and macro average precision is 0.428. Active-region cluster-bootstrap 95% intervals for those macro values are [0.662, 0.857] and [0.364, 0.577], respectively. All CNN intervals use 66 test active regions and 2000 bootstrap replicates.

GPU inference with the existing best checkpoint completed on AMD Radeon Graphics using PyTorch 2.11.0+rocm7.2 in 30.8 seconds. CPU/GPU probabilities agreed within 1.8e-7 on the first four identical test samples. The checkpoint configuration already uses focal loss (gamma 0.8), inverse-square-root class sampling, and class weights, so it is not an unweighted baseline. No controlled new imbalance variants were run in this pass.

## Binary model results

All values below are observation-level on the same held-out rows. Confidence intervals are active-region cluster-bootstrap percentile intervals. The complete 12-metric JSON per model and target includes counts, prevalence, confusion matrix, and intervals for every metric.

| Target | Model | Prevalence | Threshold | ROC-AUC (95% CI) | PR-AUC (95% CI) | TSS (95% CI) | HSS | CSI | MCC | F1 | Precision | Recall | Specificity | FAR | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ge_c | XGBoost | 0.227 | 0.2 | 0.839 (0.790-0.880) | 0.660 (0.524-0.757) | 0.527 (0.437-0.605) | 0.432 | 0.420 | 0.457 | 0.592 | 0.478 | 0.776 | 0.751 | 0.522 | 0.121 |
| ge_c | Logistic Regression | 0.227 | 0.2 | 0.846 (0.797-0.885) | 0.679 (0.548-0.767) | 0.537 (0.448-0.614) | 0.443 | 0.428 | 0.468 | 0.599 | 0.486 | 0.780 | 0.758 | 0.514 | 0.119 |
| ge_c | HistGradientBoosting | 0.227 | 0.2 | 0.834 (0.783-0.875) | 0.647 (0.508-0.747) | 0.520 (0.433-0.599) | 0.430 | 0.418 | 0.454 | 0.589 | 0.480 | 0.764 | 0.756 | 0.520 | 0.124 |
| ge_c | Extra Trees | 0.227 | 0.3 | 0.830 (0.778-0.872) | 0.632 (0.494-0.735) | 0.506 (0.406-0.591) | 0.442 | 0.419 | 0.454 | 0.590 | 0.504 | 0.711 | 0.794 | 0.496 | 0.125 |
| ge_m | XGBoost | 0.035 | 0.1 | 0.904 (0.810-0.959) | 0.372 (0.129-0.645) | 0.581 (0.325-0.731) | 0.354 | 0.239 | 0.390 | 0.385 | 0.275 | 0.642 | 0.939 | 0.725 | 0.026 |
| ge_m | Logistic Regression | 0.035 | 0.1 | 0.922 (0.847-0.968) | 0.417 (0.168-0.668) | 0.582 (0.323-0.738) | 0.402 | 0.273 | 0.426 | 0.429 | 0.326 | 0.629 | 0.953 | 0.674 | 0.025 |
| ge_m | HistGradientBoosting | 0.035 | 0.1 | 0.892 (0.783-0.956) | 0.381 (0.118-0.648) | 0.538 (0.288-0.702) | 0.383 | 0.259 | 0.403 | 0.411 | 0.317 | 0.583 | 0.955 | 0.683 | 0.027 |
| ge_m | Extra Trees | 0.035 | 0.1 | 0.902 (0.819-0.952) | 0.314 (0.113-0.612) | 0.605 (0.373-0.744) | 0.283 | 0.192 | 0.345 | 0.321 | 0.208 | 0.702 | 0.904 | 0.792 | 0.028 |

The positive prevalence is 0.227 for ge_c and 0.035 for ge_m. High ranking performance does not imply a favorable operating point: at ge_m's frozen XGBoost threshold 0.1, recall is 0.642 and FAR is 0.725, with precision 0.275. Thresholds were not changed using this sensitivity analysis.

## Paired AUC comparisons

Differences are XGBoost minus comparator, paired by active-region bootstrap on identical observation-level test rows. No p-values are reported; the paired 95% intervals are used descriptively and no inferential test claim is made.

| Target | Comparison | AUC difference | Paired 95% CI |
|---|---|---:|---:|
| ge_c | XGBoost vs Logistic Regression | -0.007 | [-0.0190, 0.0053] |
| ge_c | XGBoost vs HistGradientBoosting | 0.005 | [-0.0009, 0.0120] |
| ge_c | XGBoost vs Extra Trees | 0.009 | [0.0014, 0.0175] |
| ge_m | XGBoost vs Logistic Regression | -0.019 | [-0.0541, 0.0005] |
| ge_m | XGBoost vs HistGradientBoosting | 0.012 | [0.0007, 0.0304] |
| ge_m | XGBoost vs Extra Trees | 0.002 | [-0.0199, 0.0174] |

The paired intervals exclude zero for XGBoost versus Extra Trees on ge_c and XGBoost versus HistGradientBoosting on ge_m. The other four intervals cross zero. No p-values or multiplicity-adjusted significance claims are made.

## Calibration and baselines

Calibration uses 10 equal-width bins on [0,1]; ECE is the bin-frequency-weighted absolute difference between mean predicted probability and observed frequency. This is distinct from discrimination (ROC-AUC/PR-AUC). Brier score and ECE are in the table below; no recalibration was performed.

| Target | Model | Brier | ECE |
|---|---|---:|---:|
| ge_c | XGBoost | 0.121 | 0.019 |
| ge_c | Logistic Regression | 0.119 | 0.021 |
| ge_c | HistGradientBoosting | 0.124 | 0.025 |
| ge_c | Extra Trees | 0.125 | 0.027 |
| ge_m | XGBoost | 0.026 | 0.008 |
| ge_m | Logistic Regression | 0.025 | 0.003 |
| ge_m | HistGradientBoosting | 0.027 | 0.014 |
| ge_m | Extra Trees | 0.028 | 0.007 |

The majority-climatology baseline assigns a constant training-prevalence score, giving ROC-AUC 0.5 and average precision equal to observed test prevalence; the no-skill PR reference is also test prevalence. These baselines are saved in `results/baseline_comparison.csv`. CNN-LSTM accuracy alone is not an adequate comparison under this class imbalance.

## Threshold sensitivity

The ge_m sweep is descriptive on Test; the selected validation threshold remains 0.1. At thresholds 0.05, 0.10, 0.20, 0.30, and 0.50, XGBoost precision is 0.200, 0.275, 0.402, 0.483, 0.535 and recall is 0.778, 0.642, 0.539, 0.445, 0.260. Equivalent ge_c values are saved alongside it.

## Feature importance and ablation

Valid gain and split-frequency importance was reconstructed from the saved XGBoost JSON tree statistics because the pre-existing importance CSVs contain only zeros. Each saved model used 27 of 29 columns. Feature names are generic feature_00 through feature_28; the source contains no validated physical-name mapping. Importance is therefore shown only by generic column ID, not interpreted as gradient/PIL/wavelet/flux evidence. Feature-group ablations and SHAP were not performed: group mapping is ambiguous and XGBoost is unavailable in the fresh environment.

## Feasibility and limitations

- The primary tabular prediction files are real saved probabilities; HistGradientBoosting and Extra Trees probabilities were generated from existing fitted models, and their thresholds/point metrics were checked against existing records. No tabular model was retrained.
- The saved HistGradientBoosting/Extra Trees joblib models were serialized under scikit-learn 1.9.0 and loaded under 1.9.1, which emitted version warnings. HistGradientBoosting reproduces its stored metrics; regenerated Extra Trees ROC-AUC/AP differ from the historical metric JSON by less than 0.000004. The new saved probabilities and metrics are the values used here.
- No chronological experiment was run. Existing split timestamp ranges overlap; a separate grouped chronological protocol would require a new, predeclared split and new training runs.
- The CNN sequence test has 2,021 C/M/X examples and no no-flare class. Although all 2,021 sequence endpoints occur in the tabular test file, the trained CNN's multiclass task is not the ge_c/ge_m binary task and does not include negatives. A valid matched binary comparison is therefore not supported by the current CNN output.
- The CNN already uses imbalance-aware training. Additional controlled variants were not run in this GPU inference pass; no claim of improvement is made.
- The existing XGBoost headline AUCs 0.891 (ge_c) and 0.939 (ge_m) were active-region-aggregated results, while comparison model values were observation-level. The consistent observation-level XGBoost results are 0.839 and 0.904. The two units must not be mixed in cross-model ranking.
- The NASA/SHARP CSV remains separate and was not used. No external-validation results are claimed.
- There is no editable IEEE LaTeX source in the repository, only `Sun-Is-A-Deadly-Laser.pdf`. The compiled PDF was not altered. A manuscript revision requires the source files.

## Artifact locations

Complete predictions and metrics are under `experiments/<model>/<target>/predictions/` and `experiments/<model>/<target>/metrics/comprehensive_metrics.json`; the Phase 0 inventory is `results/PHASE0_ARTIFACT_AUDIT.md`. Statistical intervals/comparisons, calibration, baselines, threshold sweeps, and importance tables are under `results/`. Publication figures are in `diagrams/`, with PNG, PDF, and SVG versions for Figures 1-11 except Figure 8, which is available only for the two XGBoost importance panels. Figure 12 was not generated because no chronological experiment exists.

## Reproduction

Run one model-target analysis at a time, then regenerate aggregate outputs:

```bash
source .research-venv/bin/activate
python build_saved_predictions.py --target ge_c --model histgradientboosting
python build_saved_predictions.py --target ge_m --model histgradientboosting
python build_saved_predictions.py --target ge_c --model extratrees
python build_saved_predictions.py --target ge_m --model extratrees
python analyze_saved_predictions.py --target ge_c --model xgboost
python analyze_saved_predictions.py --target ge_c --model logistic_regression
python analyze_saved_predictions.py --target ge_c --model histgradientboosting
python analyze_saved_predictions.py --target ge_c --model extratrees
python analyze_saved_predictions.py --target ge_m --model xgboost
python analyze_saved_predictions.py --target ge_m --model logistic_regression
python analyze_saved_predictions.py --target ge_m --model histgradientboosting
python analyze_saved_predictions.py --target ge_m --model extratrees
python research_analysis.py
```
