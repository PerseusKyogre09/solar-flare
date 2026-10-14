# Implementation Checkpoint 3

Date: 2026-10-11

## Completed

Phase 3 preflight implementation is complete:

- Added `build_matched_population.py`.
- Generated all-label sequence manifests under `results/matched_experiment/`.
- Matched all endpoint rows to the 29-feature CSV.
- Checked image availability for every retained frame.
- Checked active-region overlap across splits.
- Preserved historical sequence manifests, models and results.

Generated files:

- `results/matched_experiment/train_all_label_sequences.csv`
- `results/matched_experiment/validation_all_label_sequences.csv`
- `results/matched_experiment/test_all_label_sequences.csv`
- `results/matched_experiment/matched_all_label_endpoint_features.csv`
- `results/matched_experiment/matched_population_summary.json`

The population builder can be reproduced with:

```bash
venv/bin/python build_matched_population.py
```

## Verified population

| Split | Sequences | Active regions | ≥M positive | 0/C negative | Missing-image sequences |
|---|---:|---:|---:|---:|---:|
| Train | 73,278 | 1,253 | 2,264 | 71,014 | 0 |
| Validation | 9,240 | 156 | 271 | 8,969 | 0 |
| Test | 9,165 | 156 | 313 | 8,852 | 0 |
| **Total** | **91,683** | — | **2,848** | **88,835** | **0** |

Matching and integrity results:

- Feature matches: 91,683 / 91,683.
- Endpoint-to-feature unmatched rows: 0.
- Active-region overlap:
  - Train/Validation: 0.
  - Train/Test: 0.
  - Validation/Test: 0.
- Candidate windows and cadence failures:
  - Train: 75,421 candidate windows; 2,143 cadence failures.
  - Validation: 9,525 candidate windows; 285 cadence failures.
  - Test: 9,410 candidate windows; 245 cadence failures.
- All retained sequences have all ten image files available locally.

## Training status

No strict matched model was trained.

The active environments report no usable accelerator:

```text
venv:          torch 2.10.0+rocm7.0, torch.cuda.is_available() = False
.research-venv: torch 2.11.0+rocm7.2, torch.cuda.is_available() = False
```

The local host therefore has no available CUDA/ROCm device for the new image branch. The strict all-label population contains 91,683 ten-frame sequences, compared with 14,053 sequences in the historical CNN train split. Launching a full binary CNN training job on CPU would be a long, unvalidated computation and is not reasonable under this environment. It was intentionally not started.

No tabular-only training was launched because the required scientific comparison is incomplete without the matched image branch, and starting one branch would create an asymmetric partial experiment.

## Outstanding implementation

Still required:

1. Implement the binary all-label CNN trainer or a declared binary adaptation of the existing architecture.
2. Implement the matched tabular trainer using only the all-label endpoint-derived training population.
3. Add validation-only checkpoint and threshold selection.
4. Export per-example validation/test probabilities and metadata.
5. Run a predeclared seed set on both branches.
6. Generate paired active-region cluster-bootstrap results.
7. Run the predeclared ablations where compute permits.
8. Create Phase 4 result tables/figures after successful paired runs.

## Historical results

No historical result was replaced or regenerated. Existing C/M/X CNN runs remain separate multiclass results. Existing tabular results remain full image-level binary results. The new all-label endpoint population is a distinct, verified population and must not be mixed with those historical metrics.

## Next execution command

After a CUDA-capable environment is available and the binary trainer is implemented:

```bash
venv/bin/python build_matched_population.py
```

Then run the frozen model-training/evaluation commands documented in `EXPERIMENT_PROTOCOL.md`.


