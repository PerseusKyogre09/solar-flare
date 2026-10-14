# Implementation Checkpoint 4

Date: 2026-10-11

## Completed

- Verified all 11 historical CNN-LSTM seed directories without retraining or rerunning evaluation.
- Recalculated per-seed metrics, sample standard deviations, minima and maxima from the saved JSON files.
- Verified that every historical result reports 2,021 test sequences and that every confusion matrix sums to 2,021.
- Ran a real-image forward/backward/optimizer smoke test using the project `venv`.
- Saved the smoke-test checkpoint and report under `results/diagnostics/`.

## Historical outputs

- `results/historical_cnn_lstm/historical_seed_metrics.csv`
- `results/historical_cnn_lstm/historical_seed_metrics.json`
- `results/historical_cnn_lstm/HISTORICAL_SUMMARY.md`

The 11-seed aggregate reproduces the previously provisional values closely: accuracy mean 0.765957 (sample SD 0.047777), macro-F1 mean 0.438094 (0.030414), macro TSS mean 0.266567 (0.066294), macro HSS mean 0.213106 (0.035071), and X recall mean 0.106952 (0.174634).

## Accelerator diagnostic

The local `venv` contains `torch 2.10.0+rocm7.0`, but this runtime reports:

```text
torch.cuda.is_available() = False
torch.cuda.device_count() = 0
rocminfo = command not found
```

The real-image smoke test therefore ran on CPU. It successfully decoded a 224x224 magnetogram, performed a forward pass, loss calculation, backward pass, optimizer step, checkpoint save/reload, and probability export. The output is `results/diagnostics/accelerator_smoke_test.json` and the checkpoint is `results/diagnostics/smoke_checkpoint.pt`.

## Stopped work

Full strict matched CNN training was not started. The all-label image population contains 73,278 training sequences and 91,683 total sequences; CPU execution is not a reasonable substitute for the required accelerator experiment. The strict binary CNN trainer, matched tabular trainer, prediction exports, ablations, bootstrap analysis and publication figures remain unfinished.

## Reproduction commands

```bash
source venv/bin/activate
venv/bin/python aggregate_historical_cnn.py
venv/bin/python accelerator_smoke_test.py
```

For the next phase, run the smoke test on a machine where ROCm exposes the Radeon device, then implement/run the strict binary branches from `EXPERIMENT_PROTOCOL.md`. Do not interpret the CPU smoke test as a completed research experiment.
