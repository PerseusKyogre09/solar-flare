# Implementation Checkpoint 2

Date: 2026-10-11

## Protocol status

Phase 2 is complete. The strict matched protocol is frozen in:

`EXPERIMENT_PROTOCOL.md`

No model training has been started for the new matched experiment.

## Critical population decision

The current CNN sequence manifests are not suitable as the strict binary ≥M training population because `make_sequences.py:46-57` removes all `0` labels before sequence construction.

The current C/M/X-only endpoint match is valid for:

- a secondary post-hoc comparison;
- deriving `P(≥M)=P(M)+P(X)` from an existing multiclass CNN;
- evaluating the existing tabular model on the same endpoint subset.

It is not a strict binary image-vs-tabular training comparison because it contains no 0-labeled endpoints and the CNN was optimized for C/M/X classification.

The strict protocol therefore requires a new all-label endpoint manifest that retains 0, C, M and X labels, uses the same ten-frame/12-minute/stride-10 construction, and maps M/X to positive and 0/C to negative.

## Verified all-label population

The read-only count check over existing split manifests and labels produced:

| Split | All-label sequences | ARs | ≥M positive | 0/C negative | Positive prevalence |
|---|---:|---:|---:|---:|---:|
| Train | 73,278 | 1,253 | 2,264 | 71,014 | 0.0309 |
| Validation | 9,240 | 156 | 271 | 8,969 | 0.0293 |
| Test | 9,165 | 156 | 313 | 8,852 | 0.0342 |
| **Total** | **91,683** | — | **2,848** | **88,835** | — |

These counts are a pre-training verification target. The implementation phase must regenerate them using a source-controlled all-label manifest builder and fail if the counts or integrity assertions differ without explanation.

## Existing endpoint audit

The Phase 1 audit generated:

- `results/matched_audit/sequence_eligibility_ledger.csv`
- `results/matched_audit/sequence_endpoint_tabular_match.csv`
- `results/matched_audit/cnn_seed_inventory.csv`
- `results/matched_audit/matched_audit_summary.json`

The existing C/M/X-only endpoint match is exact:

- 17,949 endpoints;
- 17,949 tabular feature matches;
- 0 unmatched endpoints;
- 0 target disagreements;
- train/validation/test counts 14,053 / 1,875 / 2,021;
- test ≥M count 313 positive and 1,708 negative within the C/M/X-only population.

The strict all-label endpoint population must be matched separately. Its expected test population is 9,165 endpoints with 313 positives and 8,852 negatives.

## Frozen strict experiment

Both branches will use the all-label endpoint population and identical split/endpoint keys.

### Image branch

- Binary M/X output.
- Ten chronological magnetogram frames.
- Same final-frame inherited label semantics.
- Validation-only checkpoint selection.
- Validation-only threshold selection.
- Per-example probability export.
- Declared random seeds; no best-seed selection.

### Tabular branch

- 29 engineered features matched by final endpoint filename.
- Same all-label endpoint-derived train/validation/test rows.
- Training-only preprocessing.
- Validation-only threshold selection.
- Per-example probability export.

### Secondary analysis

The existing C/M/X CNN may be evaluated with `P(M)+P(X)` on the 2,021 C/M/X-only endpoints. This will be labeled secondary and will not be called the strict matched binary experiment.

Existing tabular models may be evaluated on the matched endpoint subsets as secondary analyses. They were trained on different populations and must not be presented as strict matched retraining.

## Required assertions before training

The implementation must assert:

1. Unique endpoint filename/timestamp/AR keys within each split.
2. No endpoint key overlap across splits.
3. Pairwise-disjoint active-region IDs across splits.
4. Exact one-to-one endpoint-to-feature matching.
5. Zero inherited-label disagreements.
6. Complete image availability for every frame.
7. Expected 0/C/M/X counts.
8. Identical endpoint keys supplied to both model branches.

If any assertion fails, training stops.

## Seed and artifact requirements

The strict experiment will use a predeclared seed set recorded before training. Every run must save:

- seed;
- model/config;
- checkpoint;
- best epoch and validation selection score;
- software/hardware metadata;
- training duration;
- validation predictions/probabilities;
- test predictions/probabilities;
- threshold selected on validation;
- test metrics;
- confusion matrices;
- failure status if incomplete.

Historical files are not overwritten.

## Phase 2 decision

The repository data support a defensible strict matched ≥M experiment, but only after constructing the all-label endpoint population. The current C/M/X-only sequence files should remain unchanged and should not be repurposed.

Proceed to implementation of the all-label manifest, binary image model, matched tabular driver and pre-training assertions. Do not use test results to alter this protocol.

