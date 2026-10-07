# Solar Flare Prediction

Machine-learning experiments for solar-flare prediction from solar active-region observations. The repository compares an image-sequence CNN-LSTM with four tabular classifiers trained on engineered magnetic features.

## Models

- CNN-LSTM for multiclass C/M/X flare classification from magnetogram image sequences.
- XGBoost for binary C-or-higher (`ge_c`) and M-or-higher (`ge_m`) prediction.
- Logistic regression baseline.
- Scikit-learn HistGradientBoosting baseline.
- Extra Trees ensemble baseline.

The CNN-LSTM and tabular models use different input representations and target formulations. Their results should be interpreted as complementary experiments rather than a direct accuracy ranking.

## Dataset

The primary experiments use a Dryad-derived dataset containing 29 engineered magnetic features, flare labels, image filenames, and predefined active-region-grouped splits. The large source files are intentionally excluded from version control. Place the following files under `data/` before running the tabular pipelines:

```text
Lat60_Lon60_Nans0_C1.0_24hr_png_224_features.csv
C1.0_24hr_224_png_Labels.txt
Train_Data_by_AR_png_224.csv
Validation_Data_by_AR_png_224.csv
Test_Data_by_AR_png_224.csv
```

The NASA/SHARP-related CSV is retained as a separate resource and is not mixed with the benchmark experiments. It requires a separate preprocessing and validation pipeline.

## Results

The current test results are summarized below. Binary metrics use validation-selected thresholds.

| Model | Target | ROC-AUC | PR-AUC | TSS | F1 |
|---|---|---:|---:|---:|---:|
| XGBoost | C or higher | 0.891 | 0.853 | 0.503 | 0.749 |
| XGBoost | M or higher | 0.939 | 0.721 | 0.757 | 0.500 |
| Logistic regression | C or higher | 0.846 | — | 0.537 | — |
| Logistic regression | M or higher | 0.922 | — | 0.582 | — |
| HistGradientBoosting | C or higher | 0.834 | 0.647 | 0.520 | 0.589 |
| HistGradientBoosting | M or higher | 0.892 | 0.381 | 0.538 | 0.411 |
| Extra Trees | C or higher | 0.830 | 0.632 | 0.506 | 0.590 |
| Extra Trees | M or higher | 0.902 | 0.314 | 0.605 | 0.321 |

*All results were obtained by training on a single AMD Radeon RX 9060 XT with 16 GB of DDR6 memory.*

## Reproduction

Use Python from the repository environment:

```bash
venv/bin/python -m src.train_xgboost --target ge_c
venv/bin/python -m src.train_xgboost --target ge_m
venv/bin/python train_histgb.py --target ge_c
venv/bin/python train_histgb.py --target ge_m
venv/bin/python train_extratrees.py --target ge_c
venv/bin/python train_extratrees.py --target ge_m
```

Results are written to `experiments/xgboost/`, `experiments/histgb/`, and `experiments/extratrees/`.

## Evaluation protocol

The tabular loader checks filename alignment, malformed labels, duplicate rows, and active-region overlap between Train, Validation, and Test. Thresholds are selected on Validation using maximum TSS and then applied to the held-out Test split. The predefined timestamp ranges overlap, so the protocol is active-region-grouped evaluation rather than chronological forecasting.

Because the targets are imbalanced, ROC-AUC is reported together with PR-AUC, TSS, precision, recall, and F1. The `ge_m` recall results involve substantial false-alarm trade-offs.

## Repository contents

```text
src/                         Dataset loading, metrics, and XGBoost pipeline
experiments/                 Saved metrics, predictions, reports, and compact models
data/                        Split manifests and dataset instructions
train_cnn_lstm.py            CNN-LSTM training entry point
train_histgb.py              HistGradientBoosting training entry point
train_extratrees.py          Extra Trees training entry point
README_RESEARCH.md           Detailed research and writing notes
RESULTS_COMPARISON.md        Compact model comparison
```

## Citations

The project report is available as [The Sun Is a Deadly Laser: Solar Flare Prediction](Sun-Is-A-Deadly-Laser.pdf). Cite it as:

> P. Pal, A. A. Singh, and A. Chauhan, “The Sun Is a Deadly Laser: Solar Flare Prediction,” Department of Computer Science and Engineering, SRM Institute of Science and Technology, Ghaziabad, India, 2026.

The report describes the five-model benchmark, SDO/HMI dataset, active-region-grouped protocol, binary `≥C` and `≥M` tasks, and multiclass CNN-LSTM task. Complete setup and download instructions are in [SETUP.md](SETUP.md).

The benchmark dataset should also be cited as Boucheron et al., “Solar active region magnetogram image dataset for studies of space weather,” *Scientific Data*, 2023, DOI [10.1038/s41597-023-02628-8](https://doi.org/10.1038/s41597-023-02628-8), together with the relevant Dryad record [10.5061/dryad.jq2bvq898](https://doi.org/10.5061/dryad.jq2bvq898). Method citations are listed in the research report.

## License

Project code is licensed under the [MIT License](LICENSE). Dataset and source-code licenses from original providers remain separate.
