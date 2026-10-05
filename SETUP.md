# Setup and Data Preparation

This guide reproduces the experiments described in [README.md](README.md).

## Requirements

- Linux or macOS
- Python 3.10 or newer
- Git
- At least 16 GB RAM for full tabular runs
- Sufficient storage for the image archive and feature files

The GitHub repository contains code, reports, and compact results. Large datasets and image archives are excluded from version control.

## Installation

```bash
git clone https://github.com/PerseusKyogre09/solar-flare.git
cd solar-flare
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-xgboost.txt
python -m pip install torch torchvision pandas numpy scikit-learn scipy matplotlib joblib pytest
```

For AMD GPU training, install a PyTorch ROCm wheel compatible with the host ROCm driver. Check availability with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

PyTorch uses the `torch.cuda` API for ROCm devices. A `False` result means that the host does not expose a usable GPU to the environment. On Linux, verify access to `/dev/kfd` and `/dev/dri/renderD*`.

## Benchmark dataset

The benchmark is derived from the SDO/HMI active-region dataset described by the AR-flares project. Download the reduced-resolution PNG record from [Dryad, DOI 10.5061/dryad.dv41ns23n](https://doi.org/10.5061/dryad.dv41ns23n).

Related records are [the preconfigured dataset](https://doi.org/10.5061/dryad.jq2bvq898), [the extra-image record](https://doi.org/10.5061/dryad.qjq2bvqmj), and the [AR-flares source repository](https://github.com/DuckDuckPig/AR-flares).

Place these files under `data/` with the exact names shown:

```text
data/Lat60_Lon60_Nans0_C1.0_24hr_png_224_features.csv
data/C1.0_24hr_224_png_Labels.txt
data/Train_Data_by_AR_png_224.csv
data/Validation_Data_by_AR_png_224.csv
data/Test_Data_by_AR_png_224.csv
```

Use files from the same reduced-resolution configuration. Do not mix reduced-resolution, full-resolution, or extra-image records.

The CNN-LSTM also requires the corresponding magnetogram PNG files and sequence manifests. Local sequence files are `data/Train_sequences.csv`, `data/Validation_sequences.csv`, and `data/Test_sequences.csv`; image files belong under `data/images/` and are not versioned.

## Optional NASA SHARP data

NASA/SDO SHARP data can be accessed through the [JSOC SHARP documentation](http://jsoc.stanford.edu/HMI/LOS_Sharp_series.html) and [JSOC data access interface](http://jsoc.stanford.edu/ajax/lookdata.html). SHARP data are not used by the current benchmark. They have a different schema and require a separate preprocessing, labeling, and evaluation pipeline.

## Validate and train

```bash
venv/bin/python -m pytest -q tests/test_xgb_baseline.py
venv/bin/python -m src.train_xgboost --target ge_c --limit-per-split 200
venv/bin/python -m src.train_xgboost --target ge_m --limit-per-split 200
```

Full tabular runs:

```bash
venv/bin/python -m src.train_xgboost --target ge_c
venv/bin/python -m src.train_xgboost --target ge_m
venv/bin/python train_histgb.py --target ge_c
venv/bin/python train_histgb.py --target ge_m
venv/bin/python train_extratrees.py --target ge_c
venv/bin/python train_extratrees.py --target ge_m
```

Train and evaluate the CNN-LSTM with:

```bash
venv/bin/python train_cnn_lstm.py
venv/bin/python evaluate_cnn_lstm.py
```

Outputs are written to `experiments/xgboost/`, `experiments/histgb/`, and `experiments/extratrees/`. Existing CNN-LSTM reports and figures are stored at the project root.

## Labels and evaluation

- `ge_c`: C, M, and X are positive; `0` is negative.
- `ge_m`: M and X are positive; `0` and C are negative.

The tabular pipeline uses 29 engineered features and predefined active-region-grouped Train, Validation, and Test files. It checks filenames, labels, duplicate rows, and active-region overlap. Thresholds are selected on Validation using maximum TSS and then applied to held-out Test. Timestamp ranges overlap, so results are grouped benchmark results rather than chronological forecasting results.

Because the targets are imbalanced, report ROC-AUC together with PR-AUC, TSS, precision, recall, and F1. The CNN-LSTM is a separate multiclass C/M/X task and should be reported with macro and per-class metrics.

## Troubleshooting

If the feature file is missing, confirm that `data/Lat60_Lon60_Nans0_C1.0_24hr_png_224_features.csv` exists and was downloaded from the matching Dryad configuration. Alignment errors usually mean that files from different configurations were mixed.

Large data files, image archives, caches, and oversized Extra Trees joblib files are excluded by `.gitignore`. Download them locally rather than committing them.

## Licensing and citations

Project code is released under the MIT License in [LICENSE](LICENSE). For this project, cite P. Pal, A. A. Singh, and A. Chauhan, “The Sun Is a Deadly Laser: Solar Flare Prediction,” SRM Institute of Science and Technology, 2026; the full report is [Sun-Is-A-Deadly-Laser.pdf](Sun-Is-A-Deadly-Laser.pdf).

For the benchmark data, cite Boucheron et al., “Solar active region magnetogram image dataset for studies of space weather,” *Scientific Data*, 2023, DOI [10.1038/s41597-023-02628-8](https://doi.org/10.1038/s41597-023-02628-8), and the matching Dryad record [10.5061/dryad.jq2bvq898](https://doi.org/10.5061/dryad.jq2bvq898). Dataset terms remain those of the original providers.
