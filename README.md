# PPG Signal Generator - Python Version

**A physiologically realistic PPG (Photoplethysmogram) signal generator with advanced features for machine learning and signal processing research.**

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: Custom](https://img.shields.io/badge/License-Custom-yellow.svg)](LICENSE)

## 🌟 Features

### Core Capabilities
- **5 Pulse Morphologies**: Different PPG waveform shapes observed in clinical practice
- **4 Motion Artifact Types**: Device displacement, forearm/hand motion, poor contact
- **Physiological Realism**:
  - ✅ Perfect beat-to-beat continuity (zero discontinuity jumps)
  - ✅ Heart rate-respiratory coupling (4.5:1 ratio)
  - ✅ Multi-source physiological noise (sensor, drift, HRV)
  - ✅ Realistic FFT spectrum with broadened peaks
- **Flexible Data Generation**: Stratified sampling for balanced ML datasets

---

## 📦 Installation

### Requirements
- Python 3.7 or higher
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- PyTorch >= 1.9.0 (for ML modules)

### Setup
```bash
# Clone or extract the package
cd PPG_generation

# Install dependencies
pip install -r requirements.txt

# Verify installation
python main_ppg.py
```

---

## 🚀 Quick Start

### Generate Your First PPG Signal

```python
from ppg_generator import gen_PPG
import numpy as np

# Define RR intervals (in milliseconds)
RR = np.array([800, 850, 820, 810, 840])  # 5 heartbeats

# Generate PPG signal
PPG, peak_indices, peak_values = gen_PPG(RR, pulse_type=1, Fd=1000)

# PPG: Complete PPG signal array
# peak_indices: Sample indices where peaks occur
# peak_values: Amplitude values at peaks
```

### Run Main Demo
```bash
python main_ppg.py
```

This generates:
- `output/python_ppg_test_PPG.png` - PPG waveform visualization
- `output/python_ppg_test_RR.png` - RR interval plot
- `output/python_ppg_test_FFT.png` - Frequency spectrum **with top 4 peaks annotated**
- `output/python_ppg_test.npz` - Saved data

---

## 🎨 New Features (v1.1+)

### FFT Peak Visualization

The FFT plot now automatically detects and annotates the **top 4 frequency peaks**:

```python
# Automatically shown in output/python_ppg_test_FFT.png
# - Peak 1 (Red): Dominant frequency (usually heart rate)
# - Peak 2 (Orange): Second largest (often respiratory)
# - Peak 3 (Green): Third harmonic
# - Peak 4 (Purple): Fourth component
```

### Noise Level Control

The generator uses **minimal physiological noise** (0.1%) for clean waveforms. To adjust noise levels, modify `ppg_generator.py` line 184.

### Artifact Parameter Tuning

Control artifact frequency and duration. See `verify_artifact_params.py` for detailed parameter explanations.

---

## 📚 API Documentation

### `gen_PPG(RR, pulse_type, Fd)`

Generate PPG signal from RR intervals.

**Parameters:**
- `RR` (array): RR intervals in milliseconds
- `pulse_type` (int): Pulse morphology type (1-5)
- `Fd` (int): Sampling frequency in Hz (default: 1000)

**Returns:**
- `PPGmodel` (ndarray): Generated PPG signal
- `PPGpeakIdx` (ndarray): Peak sample indices
- `PPGpeakVal` (ndarray): Peak amplitude values

---

## 🎯 Usage Examples

### Example 1: Different Heart Rates

```python
# Slow heart rate (60 bpm)
RR_slow = np.ones(20) * 1000
PPG_slow, _, _ = gen_PPG(RR_slow, pulse_type=1, Fd=1000)
```

**Note**: Respiratory frequency automatically adjusts based on heart rate (HR:Resp ≈ 4.5:1)

### Example 2: HR-Respiratory Coupling Demo

```bash
python examples/demo_hr_resp_coupling.py
```

### Example 3: Different Pulse Types

```python
for pulse_type in range(1, 6):
    RR = np.ones(10) * 800
    # Each type has distinct morphology
    PPG, _, _ = gen_PPG(RR, pulse_type=pulse_type, Fd=1000)
```

---

## 🧪 Training Data Generation (Updated v2.4)

### Stratified Batch Generation

Use `generate_training_data.py` to create a balanced ML dataset. The v2.4 update implements **Stratified Sampling** to ensure equal representation of all conditions.

**Conditions:** 5 Pulse Types × 5 Noise Modes (Clean + 4 Artifacts) = 25 Combinations.

```bash
python generate_training_data.py --num_samples 20000 --output_dir dataset_v2
```

**Features:**
- **Stratified Sampling**: Automatically balances 25 distinct pulse/noise combinations
- **Automatic Labeling**: `pulse_type` (1-5), `rhythm` (SR/AF), `artifact_map` (segmentation masks)
- **Structure**:
    - `signals/*.npz`: Signal data
    - `labels/*.json`: Detailed metadata
    - `fft_features/*.npz`: Pre-computed FFTs

---

## 🧠 Deep Learning Workflow (v2.0 - v2.4)

### 1. Data Generation
Generates `(Signal, Mask, Label)` triplets for Multi-Task UNet training.

```bash
python generate_training_data.py --num_samples 50000 --output_dir ml_dataset_v2
```

### 2. Dataset Loading (`ml_training/dataset.py`)
The updated `PPGDataset` (v2.4) supports **Dual-Channel Input**:
- **Channel 0**: Normalized PPG Signal (Amplitude)
- **Channel 1**: First Derivative (Velocity)

It handles loading from the new stratified directory structure (`signals/` + `labels/`).

### 3. Training (`train_segmentation.py`)
Trains the Multi-Task UNet model to predict both **Waveform Class (1-5)** and **Artifact Mask (0-4)**.

```bash
# Train on GPU (supports remote logging via CSV and Tensorboard)
python train_segmentation.py --data_path ml_dataset_v2 --epochs 50 --batch_size 32
```

**Outputs:**
- `checkpoints_seg_v2/best_model.pth`
- `checkpoints_seg_v2/training_log.csv` (Excel-ready metrics)

### 4. Remote Deployment
Use the included SLURM scripts (`slurm_datagen.sh`, `slurm_train.sh`) for cluster execution on systems like `front.convergence.lip6.fr`.

---

## 🔄 Version History

### v2.4 (2025-12-19): Full English Translation & Stratified Sampling
- ✅ **Internationalization**: Translated all code comments to English and removed emojis for cleaner codebase.
- ✅ **Stratified Sampling**: `generate_training_data.py` now guarantees balanced representation of 25 Data Modes (5 Pulses x 5 Noise Conditions).
- ✅ **Dual-Channel Input**: Dataset now computes and returns signal velocity as a second channel for improved model performance.
- ✅ **Robust Logging**: `train_segmentation.py` now includes CSV logging for easier remote monitoring.

### v2.3 Stable (2025-12-18)
- ✅ **Critical Fix**: Resolved UNet channel mismatch (512 vs 384) in upsampling layers.
- ✅ **Stability**: Validated on remote A100 GPU cluster.

### v2.2 (2025-12-17)
- ✅ **Robust Data Generation**: Implemented "Chunked Saving" logic.
- ✅ **Remote Workflow**: Decoupled SLURM scripts.

### v2.1 (2025-12-11)
- ✅ **Deployment Logic**: Optimized `deploy_and_setup.sh`.
- ✅ **Namespace Fix**: Renamed `models.py` to `model_factory.py`.

### v2.0 (2025-12-04)
- ✅ **Multi-Task UNet Architecture**: Simultaneous Waveform Classification + Noise Segmentation.
- ✅ **New Dataset**: Created `generate_segmentation_data.py` for `(Signal, Mask, Label)` triplets.

### v1.1 (2024-11-27)
- ✅ **FFT Peak Annotations**: Automatic detection and labeling.
- ✅ **Noise Reduction**: Reduced to 0.1% for clean waveforms.
- ✅ **Artifact Parameter Guide**: Added `verify_artifact_params.py`.

---

## 👥 Credits

**Original MATLAB Implementation:**
- **Source**: [PhysioNet: ECG and PPG signals simulator for arrhythmia detection (v1.3.1)](https://physionet.org/content/ecg-ppg-simulator-arrhythmia/1.3.1/)
- Biomedical Engineering Institute, Kaunas University of Technology

**Python Implementation:**
- 2024 conversion with physiological enhancements.

---

## 📧 Support

For questions or issues, please refer to:
1. This README documentation
2. Example scripts in `examples/`
3. Code comments in source files

---

**Happy Signal Processing! 🚀**
