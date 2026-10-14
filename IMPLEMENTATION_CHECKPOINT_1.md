# Implementation Checkpoint 1

Date: 2026-10-11

This checkpoint covers Phase 1 only. No model training, downloading, package installation, checkpoint replacement, historical-result overwrite, or manuscript modification was performed.

## Scope and verified conclusion

The inherited AR-flares labels define a future-window target: `AR-flares/general_code/customize_dataset.py:300-328` labels a magnetogram when the qualifying flare peak is strictly after the magnetogram time and less than 24 hours later. The sequence builder inherits the final frame's label at `make_sequences.py:98-104`; this is retained and is not treated as leakage by itself.

A new read-only diagnostic script was added:

```text
audit_matched_population.py
```

It writes only new artifacts under:

```text
results/matched_audit/
```

Generated artifacts:

- `sequence_eligibility_ledger.csv`
- `sequence_endpoint_tabular_match.csv`
- `cnn_seed_inventory.csv`
- `matched_audit_summary.json`

## 1. Sequence construction and eligibility

The implementation stages are in `make_sequences.py`:

| Stage | Code | Behavior |
|---|---|---|
| Label source | Lines 4, 15–20 | Loads basename-to-flare labels |
| Split source | Lines 37–41 | Processes each predefined split separately |
| AR/basename parsing | Lines 43–44 | Splits manifest paths |
| Label/class mapping | Lines 46–47, 23–30 | Keeps C/M/X; maps `0` to no class |
| Timestamp parsing | Lines 49–55 | Parses `YYYYMMDD_HHMMSS` before `_TAI` |
| Invalid-row removal | Line 57 | Drops missing class or timestamp |
| Group/order | Lines 61–65 | Groups by AR and sorts by timestamp |
| Candidate windows | Lines 67–75 | Ten frames, stride 10 |
| Cadence check | Lines 77–96 | Requires every gap to equal 12 minutes |
| Target | Lines 98–104 | Uses the label of frame 9 |
| Output | Lines 106–120 | Saves frames, AR and target |

An AR can fail because it has only zero labels, too few valid C/M/X rows for ten frames, invalid timestamps, or no valid ten-frame window with exact 12-minute cadence. Image-file availability is checked later by `train_cnn_lstm.py:147-223`, not by `make_sequences.py`.

## 2. Eligibility counts

The audit used the existing manifests and label file in memory and wrote the ledger without changing source data.

| Split | Raw rows | Raw ARs | Unmatched labels | Invalid timestamps | Valid C/M/X rows | Candidate ARs | Candidate windows | Cadence failures | Valid sequences | Sequence ARs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 759,357 | 1,256 | 0 | 0 | 149,249 | 502 | 14,735 | 682 | 14,053 | 498 |
| Validation | 95,933 | 157 | 0 | 0 | 19,791 | 64 | 1,953 | 78 | 1,875 | 64 |
| Test | 94,757 | 157 | 0 | 0 | 21,542 | 68 | 2,122 | 101 | 2,021 | 66 |

The first filter is outcome-dependent by design: all 0-labeled observations are removed from the CNN sequence population. The resulting C/M/X sequence counts are:

| Split | C | M | X |
|---|---:|---:|---:|
| Train | 11,804 | 2,034 | 215 |
| Validation | 1,605 | 248 | 22 |
| Test | 1,708 | 279 | 34 |

Among valid C/M/X rows, later eligibility removed four train ARs and two test ARs; no validation AR with valid C/M/X rows was lost. The excluded valid rows were C-only in the available manifests:

- Train: 4 non-sequence ARs, 30 valid C rows.
- Validation: 0 non-sequence ARs.
- Test: 2 non-sequence ARs, 17 valid C rows.

This does not prove absence of timing-related selection bias because exact flare peak timestamps are not retained in the inherited label file.

## 3. Exact sequence-endpoint matching

The endpoint key is the basename of `frame_9`, plus:

- split;
- active-region ID;
- parsed timestamp;
- inherited target.

The tabular split's `class` column is a binary ≥C field, so the audit correctly did not use it for ≥M. Instead, it derived ≥M from the authoritative label file by matching the tabular filename basename:

- M/X → 1
- C/0 → 0

The matching implementation is in `audit_matched_population.py:120-145`, and the merge/report is in `audit_matched_population.py:149-182`.

Results:

| Split | Sequence endpoints | Exact feature matches | Sequence-only | Target disagreements | ≥M positive | ≥M negative |
|---|---:|---:|---:|---:|---:|---:|
| Train | 14,053 | 14,053 | 0 | 0 | 2,249 | 11,804 |
| Validation | 1,875 | 1,875 | 0 | 0 | 270 | 1,605 |
| Test | 2,021 | 2,021 | 0 | 0 | 313 | 1,708 |
| **Total** | **17,949** | **17,949** | **0** | **0** | **2,832** | **15,117** |

The exact matched test prevalence is:

```text
313 / 2,021 = 0.1549
```

This is not the 0.035 prevalence of the full image-level tabular test set. The matched endpoint population is conditional on valid C/M/X sequence construction and is therefore a distinct, substantially more positive population.

The complete row-level report is:

```text
results/matched_audit/sequence_endpoint_tabular_match.csv
```

The ledger is:

```text
results/matched_audit/sequence_eligibility_ledger.csv
```

## 4. Split integrity

The predefined tabular loader checks active-region overlap and duplicate/incomplete joins at `src/xgb_baseline/data.py:104-143`.

The endpoint audit preserves the original split assignment from each predefined split file. The sequence manifests were generated one split at a time by `make_sequences.py:37-41`, so sequence endpoints remain associated with their source split.

The matched endpoint populations have:

- Train: 498 ARs.
- Validation: 64 ARs.
- Test: 66 ARs.

No sequence endpoint was unmatched to a tabular feature row, and no target disagreement was found after deriving ≥M from the same authoritative label file.

A strict duplicate/overlap check should still be run as part of the matched training driver before any model fit, using endpoint filename and active-region ID as keys. The current audit report confirms matching and split membership but does not replace a runtime assertion in the future experiment driver.

## 5. Seed inventory

Requested seeds:

```text
11, 12, 34, 42, 56, 69, 90, 93, 104, 702, 1000
```

Available locally with complete aggregate artifacts:

```text
42, 56, 69, 90, 104, 702, 1000
```

Unavailable in this workspace and local archive listings:

```text
11, 12, 34, 93
```

The accessible `cnn_lstm_results.zip` contains only the same seven available seeds. The older `solar-flare-research-package.zip` contains no missing seed directories.

Copy the missing runs into:

```text
runs/cnn_lstm/seed_11/
runs/cnn_lstm/seed_12/
runs/cnn_lstm/seed_34/
runs/cnn_lstm/seed_93/
```

At minimum, each should contain its seed-specific checkpoint, config, history, test JSON, classification report and confusion matrix. The inventory is machine-readable at:

```text
results/matched_audit/cnn_seed_inventory.csv
```

## 6. Phase 1 decision

The existing data support an exact matched endpoint population for a ≥M comparison:

- 17,949 total endpoints.
- 14,053 / 1,875 / 2,021 by train/validation/test.
- 498 / 64 / 66 active regions by train/validation/test.
- 2,832 positive ≥M endpoints overall.
- 313 positive ≥M test endpoints.
- Zero endpoint matches missing.
- Zero target disagreements.

The population is scientifically valid for a matched comparison only if it is explicitly described as the sequence-eligible endpoint population. It must not be compared directly with the full 94,757-row tabular test population.

No model training was started in Phase 1.

## Recommended next action

Proceed to Phase 2 and freeze the protocol around the 17,949 exact endpoint matches. The strict experiment should retrain both branches on the matched endpoint populations, while retaining the existing C/M/X CNN runs as a separate multiclass result.

Before training, add runtime assertions for:

1. exact endpoint uniqueness;
2. split/active-region disjointness;
3. zero target disagreements;
4. identical train/validation/test endpoint keys for both branches;
5. validation-only threshold selection;
6. per-example probability export with filename, timestamp, AR, split, target and seed.

