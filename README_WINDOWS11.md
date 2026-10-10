# Windows 11 CNN-LSTM Setup

These instructions are only for training and evaluating the CNN-LSTM model on Windows 11 with an NVIDIA GPU.

## 1. Install prerequisites

Install Git for Windows, Python 3.10 or newer, an up-to-date NVIDIA driver, and 7-Zip.

Open PowerShell and verify:

```powershell
python --version
git --version
nvidia-smi
```

## 2. Clone the project

```powershell
git clone https://github.com/PerseusKyogre09/solar-flare.git
cd solar-flare
```

## 3. Create the Python environment

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn scipy matplotlib joblib pytest pillow
```

## 4. Check the NVIDIA driver and choose PyTorch

Run this command in PowerShell:

```powershell
nvidia-smi
```

Look for output similar to:

```text
Driver Version: 576.XX    CUDA Version: 12.8
```

The `CUDA Version` shown by `nvidia-smi` is the highest CUDA runtime supported by the installed NVIDIA driver. It is not necessarily the CUDA Toolkit installed on the computer.

Use the latest **Stable** PyTorch build supported by that driver. Do not select Preview/Nightly. The official PyTorch selector currently provides CUDA 11.8, 12.6, and 12.8 builds.

Use this mapping:

| `nvidia-smi` CUDA Version | PyTorch install option |
|---|---|
| 12.8 or higher | CUDA 12.8 (`cu128`) |
| 12.6–12.7 | CUDA 12.6 (`cu126`) |
| 11.8–12.5 | CUDA 11.8 (`cu118`) |

If the driver is older than required, update the NVIDIA driver first from:

<https://www.nvidia.com/Download/index.aspx>

Then install CUDA-enabled PyTorch using the official selector:

<https://pytorch.org/get-started/locally/>

Choose **Stable**, Windows, Pip, Python, and the CUDA option selected from the table above. Run the generated command. Examples are:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# or, for a driver supporting CUDA 12.6:
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# or, for an older compatible driver:
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Use only one of the commands. Do not run all three.

Verify the GPU:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
```

`CUDA available` must be `True`.

## 5. Download the correct dataset

Use the reduced-resolution Dryad page:

<https://datadryad.org/dataset/doi%3A10.5061/dryad.jq2bvq898>

The configuration is `Lat60_Lon60_Nans0`, `C1.0`, `24hr`, `PNG`, `224`.

Do not use the full-resolution page:

<https://datadryad.org/dataset/doi%3A10.5061/dryad.dv41ns23n>

Files without `png_224` in their names belong to the full-resolution dataset. Do not rename them.

## 6. Download and extract PNG images

Download the image archive from Zenodo:

<https://doi.org/10.5281/zenodo.7775776>

The archive should be:

```text
Lat60_Lon60_Nans0_png_224.tar.gz
```

Place it in `data` and extract it:

```powershell
tar -xzf data\Lat60_Lon60_Nans0_png_224.tar.gz -C data
```

The final image directory must be:

```text
data\images\Lat60_Lon60_Nans0_png_224\
```

## 7. Check sequence files

The project must contain:

```text
data\Train_sequences.csv
data\Validation_sequences.csv
data\Test_sequences.csv
```

If they are not included after cloning, copy these project-specific files from the project owner's supplied `data` folder.

```powershell
Test-Path data\images\Lat60_Lon60_Nans0_png_224
Test-Path data\Train_sequences.csv
Test-Path data\Validation_sequences.csv
Test-Path data\Test_sequences.csv
```

All commands should return `True`.

## 8. Train five random seeds

Open `train_cnn_lstm.py` and change only:

```python
SEED = 42
```

Use these seeds:

```text
42, 123, 456, 789, 1000
```

Run:

```powershell
python train_cnn_lstm.py
```

The script automatically creates separate folders:

```text
runs\cnn_lstm\seed_42\
runs\cnn_lstm\seed_123\
runs\cnn_lstm\seed_456\
runs\cnn_lstm\seed_789\
runs\cnn_lstm\seed_1000\
```

Each folder contains the best checkpoint, final checkpoint, training history, and configuration. No manual copying is required.

## 9. Evaluate each seed

After training a seed, set its checkpoint and run evaluation. For seed 42:

```powershell
$env:CNN_LSTM_CHECKPOINT = "runs\cnn_lstm\seed_42\cnn_lstm_best_seed_42.pth"
python evaluate_cnn_lstm.py
```

For seed 123, use:

```powershell
$env:CNN_LSTM_CHECKPOINT = "runs\cnn_lstm\seed_123\cnn_lstm_best_seed_123.pth"
python evaluate_cnn_lstm.py
```

Change the seed number for the remaining runs. Evaluation files are automatically saved in the same seed folder:

```text
cnn_lstm_test_results.json
cnn_lstm_classification_report.txt
cnn_lstm_confusion_matrix.png
```

The CNN-LSTM is a three-class C/M/X model. It does not produce separate `ge_c` and `ge_m` results.
