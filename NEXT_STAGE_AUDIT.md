# Next-Stage Target, Sequence and Matched-Experiment Audit

Date: 2026-10-11

Scope: read-only continuation of `AUDIT_REPORT.md` and the completed investigation of the local AR-flares label-generation code. No source, data, checkpoint, result, or manuscript file was modified. No training, downloading, installation, archive extraction, or expensive computation was performed.

## Executive conclusion

The original target-leakage concern should not be treated as established leakage.

The local AR-flares generator documents a valid inherited future-window label:

- A flare event is parsed using its peak time.
- An image is labeled only when the flare peak is strictly after the image time and less than 24 hours later.
- `make_sequences.py` assigns the label of the tenth frame, so the sequence target is the inherited future-window label anchored at the final input frame.

The remaining methodological issue is documentation and auditability, not automatic leakage. The exact flare peak timestamp is not retained in the reduced label file, so the realized event-time offset cannot be reconstructed from the current sequence artifacts.

The sequence eligibility audit found:

| Split | Raw observations | Raw ARs | Nonzero labeled rows | Candidate windows | Valid sequences | Sequence ARs |
|---|---:|---:|---:|---:|---:|---:|
| Train | 759,357 | 1,256 | 149,249 | 14,735 | 14,053 | 498 |
| Validation | 95,933 | 157 | 19,791 | 1,953 | 1,875 | 64 |
| Test | 94,757 | 157 | 21,542 | 2,122 | 2,021 | 66 |

All local seed artifacts remain incomplete relative to the requested 11-seed set: 7 are available and 4 are not present in the workspace or accessible local archive listings.

## 1. Sequence eligibility audit

### 1.1 Implemented filtering stages

The complete sequence construction is in `make_sequences.py`:

1. Label file selection: line 4 uses `data/C1.0_24hr_224_png_Labels.txt`.
2. Label parsing: lines 15–20 map image basenames to flare strings.
3. Split-file selection: lines 37–41 read one `*_Data_by_AR_png_224.csv` at a time.
4. Active-region extraction: lines 43–44 split the manifest path into `ar` and basename.
5. Label join: lines 46–47 map the basename to the flare label and then to C/M/X using `get_class`.
6. Timestamp extraction: lines 49–55 parse `YYYYMMDD_HHMMSS` immediately before `_TAI`.
7. Valid-row filtering: line 57 drops rows with no C/M/X class or invalid timestamp.
8. Active-region grouping and chronological ordering: lines 61–65 group by AR and sort by time.
9. Candidate-window generation: lines 67–75 use ten frames, stride 10.
10. Cadence filtering: lines 77–96 require every adjacent frame gap to equal exactly 12 minutes.
11. Target assignment: lines 98–104 assign `window[-1]["class"]`, the label attached to frame 9.
12. Output: lines 106–120 save ten frame paths and the target.

The original label generator is `AR-flares/general_code/customize_dataset.py`:

- 24-hour configuration: lines 48–59.
- Event date, peak time, NOAA AR and flare class extraction: lines 231–249.
- Peak-time construction: lines 300–305.
- Future-window duration: lines 307–308.
- Future-only condition: lines 325–328.
- Positive label writing: lines 358–361.
- Negative `0` label writing: lines 363–380.

### 1.2 Why an active region can fail to produce sequences

An AR can fail at these stages:

- It may have only zero labels. Since `get_class` returns `None` for `0`, all such rows are dropped at line 57.
- It may have no valid timestamp matching the filename regex/format at lines 50–55.
- It may have fewer than 10 remaining C/M/X-labeled rows after filtering; the candidate range at lines 67–71 then produces no windows.
- It may have ten or more labeled rows but an irregular/missing cadence. Any window with one adjacent gap not exactly 12 minutes is rejected at lines 85–96.
- Because stride is 10, valid rows are not used in overlapping windows; this is a deliberate sampling rule at lines 6–8 and 67–71.
- Image-file existence is not checked by `make_sequences.py`; missing PNGs are filtered later by the dataset class in `train_cnn_lstm.py:147-223`. The saved sequence CSV may therefore include a sequence that training later skips if a referenced image is absent.

The label-generation semantics mean that dropping `0` rows is not merely a technical filter: the CNN sequence task is deliberately restricted to sequences whose final frame has a qualifying future-window flare label. This creates a C/M/X-only population and is materially different from the tabular binary population, which includes `0` labels.

### 1.3 Read-only per-stage counts

The following counts were computed in memory from the existing split CSVs and label file using the same operations as `make_sequences.py`. No output was written.

| Split | Raw rows | Raw ARs | Label unmatched | Invalid timestamps | Valid C/M/X rows | ARs with candidate windows | Candidate windows | Invalid-cadence windows | Valid sequences | ARs with valid sequences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 759,357 | 1,256 | 0 | 0 | 149,249 | 502 | 14,735 | 682 | 14,053 | 498 |
| Validation | 95,933 | 157 | 0 | 0 | 19,791 | 64 | 1,953 | 78 | 1,875 | 64 |
| Test | 94,757 | 157 | 0 | 0 | 21,542 | 68 | 2,122 | 101 | 2,021 | 66 |

The raw label composition is:

| Split | 0 | C | M | X |
|---|---:|---:|---:|---:|
| Train | 610,108 | 125,854 | 21,141 | 2,254 |
| Validation | 76,142 | 16,975 | 2,578 | 238 |
| Test | 73,215 | 18,230 | 2,961 | 351 |

The valid C/M/X rows retained after the line-57 filter are:

| Split | C | M | X |
|---|---:|---:|---:|
| Train | 125,854 | 21,141 | 2,254 |
| Validation | 16,975 | 2,578 | 238 |
| Test | 18,230 | 2,961 | 351 |

The resulting sequence target counts are:

| Split | C | M | X |
|---|---:|---:|---:|
| Train | 11,804 | 2,034 | 215 |
| Validation | 1,605 | 248 | 22 |
| Test | 1,708 | 279 | 34 |

### 1.4 Outcome-dependent exclusion assessment

There is clear class-dependent eligibility at the first filter:

- All `0` rows are excluded by `make_sequences.py:57`.
- The CNN therefore cannot be interpreted as a flare-versus-no-flare model.
- The sequence target population is conditional on the final-frame label being C/M/X.

Among already valid C/M/X rows, most ARs with candidate windows produce at least one valid sequence:

- Train: 502 candidate ARs, 498 sequence ARs.
- Validation: 64 candidate ARs, 64 sequence ARs.
- Test: 68 candidate ARs, 66 sequence ARs.

The non-sequence ARs contain only C-labeled valid rows in the available manifests:

- Train: 4 ARs excluded after candidate-window/cadence processing, 30 valid C rows total.
- Validation: no non-sequence ARs after the valid-row stage.
- Test: 2 ARs excluded after candidate-window/cadence processing, 17 valid C rows total.

This pattern indicates that later sequence eligibility is not visibly removing M/X ARs in these manifests, but it is not proof of no selection bias. The exact flare peak times are absent from the inherited label file, so outcome timing cannot be compared between included and excluded ARs.

A full diagnostic should preserve, for every AR and candidate window:

- split;
- AR ID;
- raw row count;
- count of `0`, C, M and X labels;
- label-join failures;
- timestamp failures;
- candidate-window count;
- cadence-failure count;
- valid sequence count;
- each frame filename and timestamp;
- final-frame timestamp;
- inherited target;
- image-file availability.

The current repository does not contain this accounting table.

## 2. Reconciliation of all 11 CNN-LSTM seeds

### 2.1 Requested seed set

```text
11, 12, 34, 42, 56, 69, 90, 93, 104, 702, 1000
```

### 2.2 Accessible local artifacts

| Seed | Local path | Best checkpoint | Final checkpoint | History | Config | Test JSON | Report/PNG | Completeness |
|---:|---|---|---|---|---|---|---|---|
| 11 | Not found | — | — | — | — | — | — | Unavailable |
| 12 | Not found | — | — | — | — | — | — | Unavailable |
| 34 | Not found | — | — | — | — | — | — | Unavailable |
| 42 | `runs/cnn_lstm/seed_42/` | Yes | Yes | Yes | Yes | Yes | Yes | Complete aggregate run |
| 56 | `runs/cnn_lstm/seed_56/` | Yes | Yes | Yes | Yes | Yes | Yes | Complete aggregate run |
| 69 | `runs/cnn_lstm/seed_69/` | Yes | Yes | Yes | Yes | Yes | Yes | Complete aggregate run |
| 90 | `runs/cnn_lstm/seed_90/` | Yes | Yes | Yes | Yes | Yes | Yes | Complete aggregate run |
| 93 | Not found | — | — | — | — | — | — | Unavailable |
| 104 | `runs/cnn_lstm/seed_104/` | Yes | Yes | Yes | Yes | Yes | Yes | Complete aggregate run |
| 702 | `runs/cnn_lstm/seed_702/` | Yes | Yes | Yes | Yes | Yes | Yes | Complete aggregate run |
| 1000 | `runs/cnn_lstm/seed_1000/` | Yes | Yes | Yes | Yes | Yes | Yes | Complete aggregate run |

The accessible archive `cnn_lstm_results.zip` contains only seeds 42, 56, 69, 90, 104, 702 and 1000. The older `solar-flare-research-package.zip` contains legacy single-run artifacts and no missing seed directories. A workspace-wide search found no local files for seeds 11, 12, 34 or 93.

This does not prove those runs do not exist globally. The missing artifacts must be copied into the Codex workspace, ideally as:

```text
runs/cnn_lstm/seed_11/
runs/cnn_lstm/seed_12/
runs/cnn_lstm/seed_34/
runs/cnn_lstm/seed_93/
```

Each should include at least:

```text
cnn_lstm_best_seed_<seed>.pth
cnn_lstm_final_seed_<seed>.pth
cnn_lstm_training_history_seed_<seed>.csv
cnn_lstm_config_seed_<seed>.txt
cnn_lstm_test_results.json
cnn_lstm_classification_report.txt
cnn_lstm_confusion_matrix.png
```

### 2.3 Verification of available runs

For all 7 available JSON files:

- Test size is 2,021.
- Confusion matrices use C/M/X ordering.
- Confusion-matrix entries sum to 2,021.
- JSON aggregate metrics are internally consistent with their saved confusion matrices.
- Config files agree on sequence length 10, batch size 8, maximum epochs 20, patience 8, AdamW, focal loss, inverse-square-root sampling and class-weighted training.
- Checkpoint selection is recorded as validation score (0.7 	imes) macro-F1 (+ 0.3 	imes) balanced recall.

The runs do not contain per-example predictions/probabilities or complete environment metadata. Therefore, identical test-row identity, paired seed comparisons, calibration, Brier score and ROC/PR recomputation cannot be independently verified from these artifacts.

### 2.4 Aggregate results from available JSON files

The earlier `AUDIT_REPORT.md` contains the 7-run recalculation:

- Accuracy: mean 0.7697, sample SD 0.0283, range 0.7209–0.8011.
- Macro precision: mean 0.4305, SD 0.0302.
- Macro recall: mean 0.4776, SD 0.0683.
- Macro-F1: mean 0.4426, SD 0.0366.
- Weighted-F1: mean 0.7882, SD 0.0165.
- Macro TSS: mean 0.2745, SD 0.0791.
- Macro HSS: mean 0.2191, SD 0.0372.
- X POD: mean 0.1008, SD 0.1920, range 0–0.4211.

A valid 11-seed mean/SD cannot be calculated until JSON files for seeds 11, 12, 34 and 93 are copied into the workspace.

## 3. Matched image-versus-tabular experiment plan

### 3.1 Common target

Use the inherited label attached to each magnetogram:

- `ge_m = 1` if the attached flare label begins with M or X.
- `ge_m = 0` if the attached label is C or 0.

The target conversion already exists in `src/xgb_baseline/data.py:26-31`.

For a sequence, use the label attached to its final frame, because that is the inherited future-window label anchored to the end of the input sequence. Do not redefine it as a new maximum-over-sequence target.

The matched experiment should explicitly state:

> Given ten consecutive 12-minute magnetograms ending at time (t_9), predict whether the inherited label for the final frame indicates an M- or X-class flare peak within the subsequent documented 24-hour window.

The exact flare peak time remains unavailable in the label file, so the experiment should not claim a more precise event offset.

### 3.2 Exact matching unit

Use the final frame filename, not only AR ID or sequence index.

For every valid CNN sequence:

- matching key: `frame_9` basename;
- timestamp: parsed from `frame_9` using the filename pattern;
- active region: sequence `ar` and filename prefix;
- target: sequence `target` mapped M/X to 1 and C to 0;
- sequence input: `frame_0` through `frame_9`.

The tabular feature file contains the same image-level filename/key and the 29 feature columns. The tabular loader already parses filename, AR and timestamp in `src/xgb_baseline/data.py:34-39,121-148`.

A non-destructive join should require:

1. Exact normalized basename equality.
2. Exact active-region equality.
3. Exact timestamp equality.
4. One-to-one matching.
5. Identical target after deriving `ge_m`.
6. The same split assignment.

Any mismatch must be reported rather than silently dropped.

### 3.3 Shared population and splits

The strict matched population should be the set of CNN sequence endpoints `frame_9` that have:

- a valid sequence;
- an available tabular feature row;
- an unambiguous inherited C/M/X label;
- a valid active-region assignment;
- a valid split assignment.

This will be much smaller than 94,757 tabular test observations. It should be reported as a sequence-endpoint matched population.

Use active-region partitions from the existing predefined split manifests. Do not randomly repartition endpoints. Verify that no AR occurs in more than one split.

Existing tabular models trained on all image-level tabular rows can be reused only for a secondary evaluation on the matched endpoint subset. That would not be a strict matched training experiment because the fitted tabular model saw additional training observations not represented in the sequence endpoint population.

For a strict comparison, retrain the tabular model on the same endpoint-derived training population and evaluate on the same endpoint-derived validation/test populations. The CNN must also be trained/evaluated on the corresponding binary target or a documented binary derivation from the same multiclass output.

### 3.4 Required saved outputs

Both branches must save one row per matched example with:

```text
sample_id / filename
active_region
timestamp
split
true_target_ge_m
predicted_probability
predicted_label_at_selected_threshold
model_name
model_config
seed
checkpoint
```

The CNN additionally needs:

- sequence ID;
- frame_0 through frame_9;
- final-frame timestamp;
- multiclass probabilities C/M/X;
- derived `ge_m` probability, explicitly defined as (P(M)+P(X)) if using the multiclass model;
- predicted multiclass class.

The tabular model additionally needs:

- all feature-column names and values or a stable feature-row ID;
- preprocessing/imputation metadata;
- model family and configuration.

Do not save only aggregate confusion matrices. Per-example probabilities are required for ROC-AUC, PR-AUC, calibration, Brier score, paired comparisons and cluster bootstrap.

### 3.5 Validation-only thresholding

For each model:

1. Fit on the matched training population.
2. Generate validation probabilities.
3. Select the operating threshold on validation only, using a predeclared rule such as maximum TSS with a documented tie-break.
4. Freeze that threshold.
5. Evaluate the untouched matched test population once.
6. Do not use test threshold sweeps to select the reported operating point.

Threshold-independent metrics include ROC-AUC and PR-AUC. Threshold-specific metrics include TSS, HSS, CSI, MCC, F1, precision, recall and FAR.

### 3.6 Paired active-region cluster bootstrap

Use the matched test examples and active-region ID as the resampling unit:

- sample the same active regions with replacement for both models;
- carry all matched examples from each sampled AR together;
- calculate each model’s metric per resample;
- report paired model differences and percentile 95% intervals;
- preserve the same frozen validation-selected threshold within each model.

This extends the existing implementation in `src/research_metrics.py:84-208`, which already supports active-region cluster-weighted binary metrics. The paired comparison must be based on identical rows, not on separate test populations.

### 3.7 Reuse versus retraining

Can be reused:

- Existing reduced-resolution labels and split manifests.
- Existing tabular feature file.
- Existing active-region split validation logic.
- Existing binary metric and cluster-bootstrap utilities.
- Existing tabular fitted models only for a clearly labeled secondary endpoint-subset evaluation.

Must be retrained for a strict matched experiment:

- Tabular model if its training population must be restricted to matched sequence endpoints.
- CNN model if a binary `ge_m` head/task is required rather than post hoc multiclass derivation.
- Any ablation or matched model whose target, population or preprocessing differs.
- All downstream thresholds, predictions, metrics, bootstrap intervals and figures for the new matched task.

## 4. Final methodology decision

### Results that remain usable

Under the inherited future-window semantics, the existing CNN results remain usable as:

- a C/M/X classification experiment;
- on 2,021 valid sequences;
- using ten consecutive 12-minute frames;
- with the target inherited from the final frame’s documented future-window label;
- under active-region-grouped train/validation/test partitions;
- provided the manuscript states the inherited target semantics accurately.

The results should not be discarded solely because `make_sequences.py` assigns `window[-1]["class"]`.

### Conclusions that remain unresolved

- The exact qualifying flare peak timestamp for each labeled image is not stored in `C1.0_24hr_224_png_Labels.txt`.
- The exact realized offset from final frame to qualifying peak cannot be recovered from current sequence CSVs.
- The sequence eligibility audit cannot determine whether excluded regions differ in flare timing or event outcome beyond inherited class counts.
- Four of the 11 seed runs are not available in this workspace.
- CNN per-example probabilities and predictions are unavailable for paired/calibration/bootstrap analyses.
- The strict matched image-versus-tabular experiment has not been run.

## 5. Updated implementation priority

1. **Documentation correction:** revise the methodology to define the inherited future-window label precisely; do not call it a newly constructed fixed horizon.
2. **Sequence audit artifact:** implement the non-destructive eligibility/accounting report described above, including final-frame timestamps and exclusion reasons.
3. **Artifact recovery:** copy seed 11, 12, 34 and 93 directories into the workspace and verify their JSONs/configs/checkpoints.
4. **Prediction export:** preserve per-example CNN probabilities, labels, timestamps, AR IDs and sequence frame metadata for every seed.
5. **Matched-population construction:** build and validate the exact endpoint join between sequence manifests and tabular features.
6. **Strict matched experiment:** retrain only after the common target/population protocol is frozen.
7. **Uncertainty analysis:** apply paired active-region cluster bootstrap to matched predictions and keep seed SD separate.
8. **Manuscript revision:** update target wording, test-population accounting, seed coverage and comparability claims after artifacts are complete.

## Recommended next action

Do not retrain the existing CNN merely to address the earlier leakage warning. First recover the four missing seed artifact directories and implement the read-only sequence eligibility/population audit. Then freeze the matched (≥M) protocol and decide whether a strict matched retraining is scientifically required.
