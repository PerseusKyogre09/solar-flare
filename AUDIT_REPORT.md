# Comprehensive Pre-Revision Audit

Date: 2026-10-11

Scope: read-only audit of the repository and accessible archives against the supplied review checklist. No training, downloading, archive extraction, source modification, manuscript modification, or result regeneration was performed.

## 1. Executive summary

The paper is **not ready for revision-only editing**. The data/code pipeline and experimental record require attention first.

The most serious issue is a target-definition mismatch:

- `make_sequences.py:98-104` sets each sequence target to `window[-1]["class"]`, i.e. the label attached to the tenth and final input frame.
- `make_sequences.py:6-8,67-99` constructs ten consecutive frames at 12-minute cadence and does not construct a future target window.
- The manuscript (PDF, Dataset/Problem Formulation section, p. 3; Methods/Results narrative) says the sequence receives the flare class associated with a forecast horizon following its final frame.

The implementation therefore does not establish the claimed post-input forecast horizon. The horizon is unresolved from the repository and the current sequence builder is inconsistent with the manuscript. This is a **Critical** validity problem.

Other major findings:

- `Revision.docx` is not present. This audit uses the checklist in the task and the repository evidence.
- Only 7 of the requested 11 seed result sets are accessible in this workspace: `42, 56, 69, 90, 104, 702, 1000`. Missing: `11, 12, 34, 93`.
- The 7 accessible runs have complete checkpoint/config/history/aggregate-evaluation artifacts, but no per-example predictions or probabilities.
- Recalculation from the 7 JSON files gives accuracy 0.7697 ± 0.0283, macro-F1 0.4426 ± 0.0366, macro-TSS 0.2745 ± 0.0791, and macro-HSS 0.2191 ± 0.0372 (sample SD). These are not a valid replacement for the requested 11-seed summary.
- The provisional values are close for accuracy, macro precision, macro-F1, weighted-F1, macro-TSS and macro-HSS, but the requested 11-seed claim cannot be verified from this workspace. The accessible 7-run X recall mean is 0.1008 with SD 0.1920; this agrees directionally with the provisional description but is not an 11-run estimate.
- The CNN test set is 2,021 sequences from 66 active regions, while the tabular test manifest is 94,757 observations from 157 active regions. This is verified. The repository does not document the full per-stage exclusion accounting.
- The tabular pipeline has strong split-overlap checks and validation-only threshold selection, but the manuscript mixes active-region-aggregated XGBoost headline metrics with observation-level comparison metrics.
- The current manuscript is only available as compiled `Sun-Is-A-Deadly-Laser.pdf`; no editable TeX/DOCX source was found.
- The repository contains substantial tabular analyses, calibration, threshold sensitivity, cluster bootstrap and figures, but no completed CNN-vs-tabular matched binary experiment, CNN-without-LSTM ablation, single-frame/temporal-pooling/GRU ablation, or controlled loss/sampling ablation set.

## 2. Repository inventory

### Manuscript and review

| Item | Evidence | Audit status |
|---|---|---|
| Manuscript | `Sun-Is-A-Deadly-Laser.pdf` | Present; compiled PDF only |
| Editable manuscript source | No `.tex`, `.docx`, or manuscript source found | Missing; revision cannot be applied reproducibly to source |
| Reviewer document | `Revision.docx` not found | Missing; checklist supplied in task was used |
| Research documentation | `README.md`, `README_RESEARCH.md`, `SETUP.md`, `RESULTS_FINAL.md`, `RESULTS_COMPARISON.md` | Present |

### Dataset, split and sequence artifacts

| Artifact | Path | Evidence |
|---|---|---|
| Reduced-resolution labels | `data/C1.0_24hr_224_png_Labels.txt` | Present |
| 29-feature CSV | `data/Lat60_Lon60_Nans0_C1.0_24hr_png_224_features.csv` | Present |
| Tabular splits | `data/Train_Data_by_AR_png_224.csv`, `Validation_Data_by_AR_png_224.csv`, `Test_Data_by_AR_png_224.csv` | Present; 759,357 / 95,933 / 94,757 rows |
| PNG archive | `data/Lat60_Lon60_Nans0_png_224.tar.gz` | Present; archive was not extracted or modified during audit |
| CNN sequence manifests | `data/Train_sequences.csv`, `Validation_sequences.csv`, `Test_sequences.csv` | Present; 14,053 / 1,875 / 2,021 sequences |
| Sequence construction | `make_sequences.py` | Present; target/horizon defect described below |
| SHARP resource | `data/sharp_mx-nf-all_2010-01-01_2024-04-30.csv` plus cache scripts | Present but documented as separate/not used |

### Models, training and evaluation

- CNN-LSTM training: `train_cnn_lstm.py`.
- CNN-LSTM evaluation: `evaluate_cnn_lstm.py`, `evaluate_cnn_lstm_gpu.py`, `evaluate_current_cnn.py`.
- Tabular training: `src/train_xgboost.py`, `train_histgb.py`, `train_extratrees.py`; logistic regression is trained inside `src/train_xgboost.py`.
- Tabular data validation and targets: `src/xgb_baseline/data.py:11-31,94-170`.
- Binary metric definitions: `src/xgb_baseline/metrics.py`.
- Research bootstrap/calibration utilities: `src/research_metrics.py`.
- Aggregate analysis: `research_pipeline.py`, `research_analysis.py`.
- Saved tabular predictions/metrics: `experiments/**/predictions/`, `experiments/**/metrics/`, `results/`.
- Saved CNN aggregate outputs: root-level legacy artifacts and per-seed `runs/cnn_lstm/seed_*/`.

### Accessible CNN seed artifacts

Complete per-seed directories are present for:

`runs/cnn_lstm/seed_42/`, `seed_56/`, `seed_69/`, `seed_90/`, `seed_104/`, `seed_702/`, `seed_1000/`.

Each accessible directory contains:

- `cnn_lstm_best_seed_<seed>.pth`
- `cnn_lstm_final_seed_<seed>.pth`
- `cnn_lstm_training_history_seed_<seed>.csv`
- `cnn_lstm_config_seed_<seed>.txt`
- `cnn_lstm_test_results.json`
- `cnn_lstm_classification_report.txt`
- `cnn_lstm_confusion_matrix.png`

The archive `cnn_lstm_results.zip` contains aggregate evaluation files for those same 7 seeds only. The older `solar-flare-research-package.zip` contains a single legacy CNN run and tabular package artifacts; it does not contain the missing four requested seed runs.

Missing from the accessible workspace:

- Seeds 11, 12, 34 and 93.
- CNN per-example test predictions.
- CNN per-example probabilities.
- CNN training logs with complete epoch-by-epoch console metadata.
- Per-seed software/hardware environment snapshots.

## 3. Recalculated accessible seed results

Metrics below were read directly from `runs/cnn_lstm/seed_*/cnn_lstm_test_results.json`. POD is the saved per-class recall equivalent. Mean and SD are across the 7 accessible seeds; SD is sample standard deviation. These are seed-to-seed descriptive statistics, not confidence intervals.

| Seed | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 | C POD | M POD | X POD | Macro TSS | Macro HSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.7417 | 0.4623 | 0.5297 | 0.4754 | 0.7736 | 0.7740 | 0.6093 | 0.2059 | 0.3452 | 0.2555 |
| 56 | 0.7892 | 0.4176 | 0.4452 | 0.4271 | 0.8013 | 0.8554 | 0.4803 | 0.0000 | 0.2420 | 0.2085 |
| 69 | 0.8011 | 0.4107 | 0.4249 | 0.4170 | 0.8034 | 0.8841 | 0.3907 | 0.0000 | 0.2016 | 0.1902 |
| 90 | 0.7209 | 0.4844 | 0.6116 | 0.5117 | 0.7585 | 0.7471 | 0.5878 | 0.4211 | 0.4211 | 0.2852 |
| 104 | 0.7838 | 0.4197 | 0.4521 | 0.4298 | 0.7991 | 0.8437 | 0.5125 | 0.0000 | 0.2539 | 0.2122 |
| 702 | 0.7724 | 0.4077 | 0.4396 | 0.4172 | 0.7878 | 0.8349 | 0.4839 | 0.0000 | 0.2254 | 0.1866 |
| 1000 | 0.7788 | 0.4112 | 0.4401 | 0.4203 | 0.7935 | 0.8437 | 0.4767 | 0.0000 | 0.2321 | 0.1954 |
| **Mean** | **0.7697** | **0.4305** | **0.4776** | **0.4426** | **0.7882** | **0.8261** | **0.5059** | **0.1008** | **0.2745** | **0.2191** |
| **Sample SD** | **0.0283** | **0.0302** | **0.0683** | **0.0366** | **0.0165** | **0.0481** | **0.0738** | **0.1920** | **0.0791** | **0.0372** |
| **Minimum** | **0.7209** | **0.4077** | **0.4249** | **0.4170** | **0.7585** | **0.7471** | **0.3907** | **0.0000** | **0.2016** | **0.0000** |
| **Maximum** | **0.8011** | **0.4844** | **0.6116** | **0.5117** | **0.8034** | **0.8841** | **0.6093** | **0.4211** | **0.4211** | **0.5000** |

The requested provisional 11-seed values were accuracy 0.766 ± 0.048, macro precision 0.429 ± 0.025, macro recall 0.474 ± 0.057, macro-F1 0.438 ± 0.030, weighted-F1 0.785 ± 0.031, macro-TSS 0.267 ± 0.066, macro-HSS 0.213 ± 0.035, and X recall approximately 0.107 ± 0.175. The accessible 7-run values are close but differ in several SDs, especially macro-TSS (0.079 vs 0.066) and macro-F1 (0.037 vs 0.030). No conclusion about the full 11-run set is possible without the missing artifacts.

### Variability sources

The artifacts support separating:

1. Seed-to-seed variability: the table above.
2. Tabular active-region cluster-bootstrap uncertainty: implemented for binary tabular predictions in `src/research_metrics.py:84-208` and saved under `results/statistical_uncertainty/`.
3. Threshold-selection uncertainty: validation threshold selection is implemented for tabular models in `src/train_xgboost.py:124-139`; test threshold sweeps are descriptive in `research_pipeline.py`.

The CNN seed JSONs do not contain bootstrap intervals, threshold-selection intervals, per-example probabilities, or paired seed comparisons. These uncertainty sources must not be combined.

## 4. Methodology audit

### A. Target and forecast horizon — Critical

**Verified implementation**

- Labels are parsed from `data/C1.0_24hr_224_png_Labels.txt` in `make_sequences.py:15-20`.
- C/M/X class is obtained by flare-string prefix in `make_sequences.py:23-30`.
- Ten frames are selected in `make_sequences.py:67-76`.
- Consecutive timestamps must differ by exactly 12 minutes in `make_sequences.py:77-96`.
- The target is explicitly the final frame: `make_sequences.py:98-104`.
- The training model has three output classes `C/M/X` in `train_cnn_lstm.py` and `evaluate_cnn_lstm.py:44-45`.
- Tabular targets are exactly `ge_c={C,M,X}` and `ge_m={M,X}` in `src/xgb_baseline/data.py:11-31`.

**Unresolved/incorrectly described**

No future target window is created in the sequence builder. Ten frames at 12-minute cadence span 108 minutes from the first to the tenth frame; they do not establish a post-tenth-frame prediction horizon. The manuscript's statement that the label is associated with a forecast horizon following the final frame is unsupported by the current code and appears contradicted by it.

The exact timestamp anchor and future horizon therefore cannot be established. The raw label semantics and dataset README needed to resolve whether each frame's attached label itself represents a future flare window were not used to document a separate target-construction rule in the repository.

**Required evidence/fix**

Provide the authoritative dataset label definition, explicitly state whether the attached label is already a future-window label, or reconstruct labels from timestamps and a declared horizon. Then regenerate sequence manifests if needed and align manuscript/code terminology.

### B. Split integrity and leakage — Mixed; tabular mostly verified, CNN incomplete

**Verified**

- Tabular loader parses active-region IDs and timestamps from filenames in `src/xgb_baseline/data.py:34-39,94-101`.
- It checks pairwise active-region overlap and rejects overlap at lines 104-134.
- It rejects duplicate split filenames and incomplete feature/split joins at lines 135-143.
- It applies feature/label joining after validation and derives targets afterward at lines 119-147.
- Tabular XGBoost thresholds are selected from validation probabilities and then applied to test at `src/train_xgboost.py:124-139`.
- The configured threshold grid is 0.1–0.9 in `src/xgb_baseline/config.json:6-10`.
- The tabular imputer, where used, is fitted on training data in `src/train_xgboost.py:97-99`.
- Sequence manifests are already partitioned by the predefined `*_Data_by_AR_png_224.csv` files before sequence construction in `make_sequences.py:37-47`.
- Actual manifests have disjoint active-region sets by construction in the supplied split lists: train 1,256 ARs, validation 64 sequence ARs / 157 image-manifest ARs, test 66 sequence ARs / 157 image-manifest ARs. The supplied sequence files contain only eligible subsets.

**Not fully verified**

- No repository check proves every generated sequence frame belongs to the same split beyond the builder's use of one split file at a time.
- No code establishes a future target window after frame 9.
- The sequence builder uses stride 10 and discards irregular windows; it does not produce an exclusion ledger.
- CNN normalization is fixed to mean/std 0.5 in `train_cnn_lstm.py`; this is not learned from training data, but the manuscript should state it.
- The training code seeds Python, NumPy, Torch, CUDA and the sampler generator at `train_cnn_lstm.py:76-95,252-275`. The handling of multi-worker seeds is platform-dependent; Windows uses `num_workers=0` at line 131, while Linux uses 4 workers. No worker-seed audit artifact is saved.
- Best-checkpoint selection uses validation score `0.7*macro-F1 + 0.3*balanced recall` at `train_cnn_lstm.py:515-572`; test data are not used for checkpoint selection in that script.
- The evaluation script loads a selected best checkpoint and evaluates the fixed test sequence manifest; it saves aggregate results only.

### C. CNN test-population discrepancy — Verified counts; exclusion cause unresolved

The actual artifacts confirm:

| Population | Rows/sequences | Active regions |
|---|---:|---:|
| Tabular test manifest | 94,757 | 157 |
| CNN test sequence manifest | 2,021 | 66 |
| Difference | 92,736 observations; 91 ARs not represented in CNN sequence file | 91 ARs |

The 91-AR difference agrees with the review premise. The code explains possible mechanisms—at least 10 frames, exact 12-minute gaps, valid class and timestamp parsing, and stride windows—but does not produce per-stage counts or identify whether exclusions depend on labels/outcomes. The sequence builder's `dropna(subset=["class","time"])`, exact-gap rejection and `len(rows)-SEQUENCE_LENGTH+1` condition are the relevant filters at `make_sequences.py:57-99`.

Selection bias cannot be ruled out. Required next artifact: an accounting table by split and active region with counts after filename join, label parse, timestamp parse, exact-cadence filtering, valid 10-frame windowing, and final target distribution, plus an outcome/label comparison between included and excluded regions.

### D. Repeated CNN seeds — Partially verified

Seven accessible runs have equivalent-looking configs: sequence length 10, batch size 8, 20 maximum epochs, patience 8, AdamW, focal loss plus inverse-square-root sampling and class weights, and validation score for best checkpoint. Their saved JSONs contain aggregate test results and confusion matrices.

Four requested seeds are absent from the workspace: 11, 12, 34 and 93. The audit cannot verify their independence or artifacts.

All accessible confusion matrices sum to 2,021 and use fixed class ordering C/M/X in `evaluate_cnn_lstm.py:428-469`. They are internally consistent with the saved aggregate metrics. However, no per-example predictions/probabilities are present, so same-example identity, paired comparisons, multiclass AUC/PR, calibration, Brier score, and pooled error analysis across seeds cannot be recomputed.

Training is independently seeded at `train_cnn_lstm.py:85-95`; the seed is saved in each config. Checkpoint selection is validation-based. The current repository does not save full environment/hardware metadata per run.

### E. Matched image-versus-tabular experiment — Missing

The current image task has 2,021 C/M/X sequences and no non-flare class. The tabular task has 94,757 image-level observations and binary labels on all classes including 0. The existing code does not create matched row-level binary labels and predictions for the same eligible population. Deriving ≥M from a multiclass prediction would not repair the sample-population mismatch.

Direct comparison is therefore not scientifically fair. A matched experiment requires:

1. An authoritative, shared target/horizon definition.
2. A common active-region split.
3. A common eligibility rule or explicit inclusion/exclusion accounting.
4. Image-derived binary labels/predictions and tabular predictions on the same test examples.
5. Validation-only selection and thresholding for both branches.
6. Paired per-example predictions and cluster-bootstrap differences.

No training should be claimed until these are implemented and run.

### F. CNN/LSTM and loss/sampling ablations — Missing

No completed valid runs were found for:

- Single-frame CNN.
- CNN without LSTM.
- Mean temporal pooling.
- CNN+GRU.
- Cross-entropy versus focal loss.
- With versus without class weights.
- With versus without inverse-square-root sampling.

The code does show one imbalance-aware configuration: focal loss, inverse-square-root sampling and class weights (`train_cnn_lstm.py`, config output). That is not an ablation.

Minimal useful ablation plan:

1. Current CNN-LSTM configuration.
2. Same architecture with focal loss replaced by cross-entropy.
3. Same architecture without class weights/sampling.
4. Single-frame CNN or temporal mean-pooling control.
5. Repeat each condition over at least 3 seeds, preferably the final seed set.

Every condition must use the same split, target, checkpoint rule and test evaluation.

### G. Statistical uncertainty — Tabular implementation present; CNN coverage incomplete

The active-region cluster bootstrap is implemented in `src/research_metrics.py:84-208`:

- Unique active regions are sampled with a multinomial bootstrap.
- All rows from a sampled region receive the same multiplicity.
- Threshold metrics, Brier, ROC-AUC and PR-AUC are computed from weighted rows.
- Percentile intervals use 2.5th and 97.5th percentiles at lines 197-208.
- Tests cover weighted rank metrics, threshold metrics, reproducibility and multiclass bootstrap utilities in `tests/test_research_bootstrap.py`.

The tabular saved outputs contain 2,000-resample cluster-bootstrap JSON files for XGBoost, logistic regression, HistGradientBoosting and Extra Trees for both targets. Paired AUC comparisons are saved under `results/statistical_comparisons/auc_comparisons.csv`.

Limitations:

- The saved CNN seed JSONs contain no per-example predictions/probabilities or cluster-bootstrap intervals.
- Exact class-recall intervals for each CNN seed cannot be independently recomputed from aggregate confusion matrices alone unless a suitable cluster-resampling implementation is applied to retained per-example labels/predictions; those files are absent.
- The audit did not calculate new bootstrap estimates.
- The manuscript must distinguish seed SD from cluster-bootstrap 95% intervals and threshold-selection sensitivity.

### H. Baselines and headline metrics — Partially verified; presentation defect

The tabular test prevalence is documented as approximately 0.227 for `ge_c` and 0.035 for `ge_m` in `research_analysis.py:716` and `RESULTS_FINAL.md`; the underlying tabular test manifest has 94,757 rows. The baseline files and calibrated metrics are present.

The code/reports include:

- Majority climatology baseline in `src/train_xgboost.py:129-139`.
- ROC-AUC, PR-AUC, TSS, HSS, CSI, MCC, F1, precision, recall, specificity, FAR and Brier in `src/research_metrics.py:12-25` and `src/xgb_baseline/metrics.py`.
- Validation-only threshold selection and threshold sensitivity.

Critical comparability issue: the manuscript's headline XGBoost values are active-region-aggregated summaries, while other comparison values and tables may be observation-level. `results/PHASE0_ARTIFACT_AUDIT.md` explicitly records this defect and gives observation-level recalculations. The final manuscript must choose one primary unit and label the other as a separate analysis.

The no-skill PR-AUC baseline should be shown directly beside PR-AUC values, not only described in prose. MCC should be primary for binary thresholded results. Threshold-specific metrics must be clearly separated from threshold-independent AUCs.

### I. Feature mapping and interpretability — Generic only

The feature loader assigns generic names `feature_00` through `feature_28` when the legacy CSV has no header, at `src/xgb_baseline/data.py:55-68`. The raw feature file confirms 29 anonymous numeric columns followed by labels and filename. No checked-in schema maps each index to a verified physical quantity.

The code computes XGBoost gain/frequency importance in `src/train_xgboost.py:120-123`; reconstruction and reporting are in `research_analysis.py:625-657`. `src/research_metrics.py:64-81` deliberately refuses physical grouping when names are generic. No SHAP implementation or completed SHAP analysis was found.

Physical claims about gradients, PILs, wavelets or flux should therefore be presented only as dataset-level provenance/group descriptions if independently documented, not as verified mappings for individual generic columns. SHAP is not supportable for physically interpreted claims until a schema mapping is recovered.

### J. Chronological generalization — Not present

The split loader records timestamp ranges but only enforces active-region disjointness. The manuscript itself states that timestamp ranges overlap and calls the result active-region-grouped rather than chronological. No chronological train/validation/test experiment or out-of-time artifact was found.

The paper should consistently say that results demonstrate generalization to unseen active regions under the predefined grouped partition. It should not call the results validated future-time forecasting until a chronological protocol is run.

## 5. Manuscript audit against the review checklist

Severity: Critical = validity; High = central missing analysis/reporting; Medium = important reporting/reproducibility; Low = presentation/optional.

| # | Review item | Severity | Status | Evidence and exact gap | Recommended action |
|---:|---|---|---|---|---|
| 1 | Scientific title | Medium | Implemented but not fully verified | PDF title is “The Sun Is a Deadly Laser: Solar Flare Prediction”; scientific scope is present but title is rhetorical | Use a precise scientific title |
| 2 | Abstract | High | Needs revision | PDF abstract reports one CNN run and tabular results; does not transparently foreground target/horizon uncertainty or 11-seed availability | Rewrite around verified question, population, limitations and repeated-seed status |
| 3 | Novelty/research gap | High | Partial | Related work exists, but novelty should not imply superiority; matched experiment absent | State contribution as controlled complementary benchmark, not head-to-head superiority |
| 4 | Research Questions | Medium | Missing/unclear | No clearly labeled Research Questions subsection found in PDF text | Add explicit questions tied to completed analyses |
| 5 | Related Work organization | Medium | Partial | PDF has physics/features, CNN/LSTM and verification discussion | Add calibration, uncertainty and rare-event verification structure |
| 6 | Dataset provenance/DOI | High | Implemented | PDF cites Boucheron/Dryad; repository links and setup docs present | Verify exact reduced-resolution DOI and file release in final source |
| 7 | Target definitions/horizon | Critical | Contradicted/unresolved | `make_sequences.py:98-104` final-frame target; manuscript claims post-final forecast horizon | Resolve label semantics, rebuild or correct manuscript |
| 8 | Dataset/split description | High | Partial | Counts and active-region grouping are documented; CNN eligible subset is not fully accounted | Add sequence eligibility table and AR counts |
| 9 | Model/preprocessing methods | Medium | Partial | Architecture/training code exists; environment/version metadata incomplete | Add exact architecture, normalization and package versions |
| 10 | Leakage controls/sequence construction | Critical | Partial | Split grouping and cadence checks exist; future target separation not established | Fix/verify temporal target and add automated audit |
| 11 | Repeated-seed results | High | Partial | 7/11 accessible; aggregate metrics available | Recover missing 4 runs and report mean/SD with seed list |
| 12 | CNN/LSTM ablations | High | Missing | No completed ablations | Run minimal ablation matrix |
| 13 | Matched image/tabular results | High | Missing | Different populations/tasks | Build shared binary task or remove comparative implication |
| 14 | Binary ≥C/≥M results | High | Implemented but unit-mixed | Targets/code/results exist; active-region vs observation units mixed | Correct tables and explicitly separate units |
| 15 | Cluster-bootstrap/paired comparisons | High | Partial | Tabular 2,000 cluster bootstrap and paired AUC artifacts exist; CNN seed uncertainty absent | Report cluster unit/CI method and add paired metrics where inputs exist |
| 16 | Class recall CIs | High | Missing for CNN | No per-example CNN predictions; aggregate confusion matrices only | Save predictions and compute region-aware recall intervals |
| 17 | Climatology PR-AUC | Medium | Partial | Baseline code/results exist; prominence in manuscript needs verification | Put no-skill PR baseline beside every PR-AUC table |
| 18 | MCC/threshold interpretation | Medium | Partial | MCC code/results exist; rare-event FAR/precision discussion exists | Make MCC primary and distinguish threshold-independent metrics |
| 19 | Threshold sensitivity/cost | Medium | Partial | Threshold sweep exists for tabular models; no cost-sensitive analysis | Retain validation-selected threshold and add optional cost analysis |
| 20 | Calibration/Brier/ECE | Medium | Implemented for tabular | `research_analysis.py` and `results/calibration/` exist; CNN calibration absent | Report tabular calibration; do not imply CNN calibration |
| 21 | Feature mapping/explainability | High | Exploratory only | Generic feature IDs; no verified mapping or SHAP | Remove physical feature claims or recover schema before SHAP |
| 22 | Literature comparison | Medium | Partial | `results/literature_comparison.csv` includes caveats and missing values | Keep explicit non-comparability caveats; do not rank across tasks |
| 23 | Figure numbering/captions/references | Medium | Partial | Figures 1–11 exist in PNG/PDF/SVG; PDF references figures | Cross-check final captions against regenerated source; editable source missing |
| 24 | Figure redundancy/quality | Low/Medium | Partial | ROC/PR curves, dot-whiskers and confusion/calibration figures exist | Remove redundant views and ensure unit/target labels |
| 25 | Computational cost | Low | Missing/partial | GPU inference metadata exists; training cost per seed not standardized | Report hardware, training time, data-loader mode and memory |
| 26 | Hardware/software reproducibility | High | Partial | `SETUP.md`, environment files and some environment reports; per-seed metadata absent | Freeze exact dependency versions and per-run hardware/config |
| 27 | Code/data availability | High | Partial | Code/data documentation exists; large data excluded and archive availability is external | State exact DOI/archive and required local filenames |
| 28 | Reproducibility table | Medium | Missing | No consolidated table found | Add table of data, split, seed, config, checkpoint and metric artifacts |
| 29 | Limitations/external validation | High | Partial | PDF acknowledges grouped/nonchronological, small X class and no matched task; SHARP separate | Add target-horizon and CNN eligibility limitations explicitly |
| 30 | Conclusion alignment | High | Partial | Conclusion is generally cautious but existing headline claims precede unresolved horizon and 11-seed coverage | Rewrite after pipeline and seed audit are resolved |

## 6. Figures and traceability

| Figure/artifact | Exists | Traceability | Audit note |
|---|---|---|---|
| `diagrams/figure1_workflow.*` | Yes | Code/report narrative | Verify architecture labels against current code |
| `figure2_class_distribution.*` | Yes | `research_analysis.py:117-124` | CNN distribution is sequence-level; tabular distributions are image-level |
| `figure3_confusion_matrix.*` and normalized | Yes | Saved tabular/CNN aggregate artifacts | Confirm which legacy CNN run it represents |
| `figure4_roc_curves.*` | Yes | `research_analysis.py` saved predictions | Unit mixing must be corrected |
| `figure5_pr_curves.*` | Yes | Saved tabular probabilities | Add no-skill prevalence line |
| `figure6_threshold_sensitivity.*` | Yes | `research_pipeline.py:94-103` | Test sweep is descriptive; validation threshold remains frozen |
| `figure7_calibration.*` | Yes | `research_pipeline.py:105-115` | Tabular only; 10 fixed-width bins |
| `figure8_feature_importance_ge_c/ge_m.*` | Yes | XGBoost JSON/reconstruction | Generic feature IDs only |
| `figure9_roc_auc_dotwhisker.*` | Yes | Saved tabular predictions/bootstrap | Check comparison unit |
| `figure10_pr_auc_dotwhisker.*` | Yes | Saved tabular predictions/bootstrap | Check comparison unit |
| `figure11_tss_dotwhisker.*` | Yes | Saved tabular metrics/bootstrap | Check comparison unit |
| Per-seed CNN matrices | Yes for 7 accessible seeds | Per-seed JSON and PNG | Four requested seed runs absent |
| Per-seed CNN ROC/PR/calibration | No | No probabilities | Cannot be reconstructed from aggregate JSON |

The PDF is compiled and the editable source is absent, so exact manuscript-to-figure source traceability is blocked. The repository contains both legacy root CNN artifacts and newer per-seed artifacts; captions must identify which run or aggregate they represent.

## 7. Reproducibility audit

### Implemented and verified

- Reduced-resolution dataset files and image archive are present locally.
- Tabular active-region split overlap checks are implemented.
- Binary target conversion is explicit.
- Tabular validation-only threshold selection is implemented.
- Tabular active-region cluster bootstrap and paired AUC comparison artifacts exist.
- CNN seeds are explicitly set in training code and recorded in accessible per-seed configs.
- CNN best checkpoint selection is validation-based.
- CNN aggregate confusion matrices and metrics are internally consistent for 7 accessible runs.
- Windows data-loader failure was addressed with `num_workers=0` on Windows at `train_cnn_lstm.py:128-131`.

### Implemented but not fully verified

- Sequence split integrity is plausible because sequence construction reads one predefined split at a time, but no automated sequence-level audit is saved.
- Reproducibility of multi-worker training across platforms is not fully documented.
- Tabular feature mapping is technically consistent but physically anonymous.
- Evaluation output separation by checkpoint path is implemented in `evaluate_cnn_lstm.py:32-35,828-951`, but old root-level legacy outputs coexist.

### Missing or blocked

- Reviewer document `Revision.docx`.
- Editable manuscript source.
- Four requested seed result sets: 11, 12, 34, 93.
- Per-example CNN predictions/probabilities for all seeds.
- Exact target forecast horizon and authoritative label-window rule.
- Sequence exclusion accounting.
- CNN cluster-bootstrap intervals and paired seed/model comparisons.
- Completed CNN architecture/loss/sampling ablations.
- Matched image/tabular binary experiment.
- Chronological out-of-time experiment.
- Verified physical mapping for generic feature columns.
- Per-seed full environment/hardware metadata.

## 8. Prioritized action plan

### Must fix before submission

1. Resolve the target/horizon contradiction. Do not describe final-frame labels as a future forecast until the dataset label semantics and code are reconciled.
2. Produce an auditable sequence eligibility/exclusion table by split and active region.
3. Recover or explicitly remove the 11-seed claim; the current workspace verifies only 7 seeds.
4. Save per-example CNN test predictions and probabilities with active-region ID, timestamp/frame endpoints, target and predicted class.
5. Reconcile manuscript tables so all headline binary comparisons use the same unit.
6. Correct CNN test-population wording: 2,021 sequences from 66 ARs are not the 94,757-observation/157-AR tabular test population.
7. Add exact environment and hardware metadata and an editable manuscript source.

### Highest-value additional experiments

1. Re-run the corrected CNN target pipeline over all declared seeds.
2. Run a matched binary ≥M image-versus-tabular experiment on identical eligible examples and splits.
3. Run current-vs-cross-entropy, current-vs-no-imbalance-control, and single-frame/temporal-pooling controls.
4. Compute CNN active-region cluster-bootstrap intervals and paired differences using saved probabilities/predictions.
5. Run a predeclared chronological split if future-time forecasting is a central claim.

### Manuscript-only revisions after validity fixes

- Replace “forecast horizon following the final frame” unless verified.
- State active-region-grouped, nonchronological evaluation consistently.
- Report seed mean/SD separately from bootstrap CIs.
- Show no-skill PR baseline and MCC.
- Limit physical feature interpretation to verified schema.
- Add sequence exclusion and test-population accounting.
- Explain X-class instability and avoid accuracy-led claims.

### Optional improvements

- Add calibration for CNN probabilities.
- Add operational cost-sensitive threshold analysis.
- Add training-time and memory reporting.
- Reduce redundant ROC/PR/dot-whisker figures.
- Add a reproducibility table and machine-readable manifest.

## 9. Exact files likely to change in the implementation phase

Pipeline/data:

- `make_sequences.py`
- `data/Train_sequences.csv`
- `data/Validation_sequences.csv`
- `data/Test_sequences.csv`
- A new non-destructive sequence-audit/accounting script
- `src/xgb_baseline/data.py` only if target/schema corrections require it

CNN training/evaluation:

- `train_cnn_lstm.py`
- `evaluate_cnn_lstm.py`
- `evaluate_cnn_lstm_gpu.py`
- A new prediction export and CNN bootstrap/aggregation script
- Per-seed run directories and environment manifests

Tabular/research analysis:

- `src/research_metrics.py`
- `research_pipeline.py`
- `research_analysis.py`
- `build_saved_predictions.py`
- `analyze_saved_predictions.py`

Manuscript/documentation:

- Missing editable manuscript source to be recovered.
- `README_RESEARCH.md`, `SETUP.md`, `RESULTS_FINAL.md`, and `RESULTS_COMPARISON.md`.
- Figure-generation source/captions after the manuscript source is recovered.

## 10. Questions unanswered from available artifacts

1. Does the attached Dryad label encode a future flare window, and if so what exact horizon and timestamp anchor does it use?
2. If it does not encode a future window, why does the manuscript describe a post-final-frame forecast horizon?
3. Why exactly are the 91 test active regions absent from the sequence test population at each filtering stage?
4. Are the missing seeds 11, 12, 34 and 93 stored outside this workspace, and do they use the same code/configuration?
5. Were all 11 runs trained independently, or were any checkpoints reused?
6. Do the missing runs include per-example predictions/probabilities or only aggregate reports?
7. What exact PyTorch, CUDA/ROCm, Python, torchvision, scikit-learn and driver versions produced each seed?
8. Are the root-level legacy CNN artifacts from the same code/configuration as the 7 per-seed runs?
9. What authoritative feature schema maps `feature_00`–`feature_28` to physical quantities?
10. What is the intended primary evaluation unit for binary metrics: image observation or active-region aggregate?
11. Is the paper's headline CNN value from the legacy run or one of the current seed runs?
12. What editable manuscript source should be revised?

## Final recommendation

The paper requires **code/data-pipeline and experimental clarification before revision-only editing**. The target/horizon definition is the blocking issue. Once that is resolved, the missing seed artifacts, sequence exclusion accounting, per-example CNN predictions, and unit-consistent binary reporting should be addressed before final manuscript revisions.
