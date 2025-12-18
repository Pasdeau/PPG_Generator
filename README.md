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
- **Flexible Data Generation**: Synthetic or real RR intervals, batch generation for ML training

---

## 📦 Installation

### Requirements
- Python 3.7 or higher
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0

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

## 🎨 New Features (v1.1)

### FFT Peak Visualization

The FFT plot now automatically detects and annotates the **top 4 frequency peaks**:

```python
# Automatically shown in output/python_ppg_test_FFT.png
# - Peak 1 (Red): Dominant frequency (usually heart rate)
# - Peak 2 (Orange): Second largest (often respiratory)
# - Peak 3 (Green): Third harmonic
# - Peak 4 (Purple): Fourth component
```

Each peak shows:
- Frequency in Hz
- Equivalent heart rate in bpm
- Magnitude value
- Color-coded markers and annotations

### Noise Level Control

The generator uses **minimal physiological noise** (0.1%) for clean waveforms:

```python
# In ppg_generator.py
hf_noise_level = 0.001 * signal_range  # Very subtle sensor noise
```

**To adjust noise levels**, modify `ppg_generator.py` line 184:
- `0.001` = Clean (current, matches MATLAB)
- `0.005` = Moderate noise
- `0.01` = Realistic clinical noise

### Artifact Parameter Tuning

Control artifact frequency and duration:

```python
# In main_ppg.py
typ_artifact = np.array([1, 1, 1, 1])  # Equal probability for all 4 types
dur_mu0 = 15  # Artifact-free interval (seconds)
dur_mu = 2    # Artifact duration (seconds)
```

**Artifact intensity levels**:

| Level | dur_mu0 | dur_mu | Artifact % | Use Case |
|-------|---------|--------|------------|----------|
| Light | 15 | 2 | ~12% | Clean training data |
| Medium | 10 | 5 | ~33% | Balanced dataset |
| Heavy | 5 | 10 | ~67% | Stress testing |

**Artifact type selection**:
```python
# Only device displacement
typ_artifact = np.array([1, 0, 0, 0])

# Device + forearm motion
typ_artifact = np.array([1, 1, 0, 0])

# All types with different weights
typ_artifact = np.array([2, 1, 1, 1])  # Device 2x more likely
```

Run `python verify_artifact_params.py` to see detailed parameter explanations.

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

**Example:**
```python
# Normal sinus rhythm
RR = np.ones(50) * 800  # 75 bpm
PPG, peaks_idx, peaks_val = gen_PPG(RR, pulse_type=1, Fd=1000)
```

---

### `gen_PPGpulse(t, pulse_type, target_baseline=None)`

Generate a single PPG pulse waveform.

**Parameters:**
- `t` (array): Normalized time vector (0 to 1)
- `pulse_type` (int): Pulse type (1-5) or parameter dict
- `target_baseline` (float, optional): Target baseline for beat continuity

**Returns:**
- `P` (ndarray): PPG pulse waveform

**Example:**
```python
from ppg_pulse import gen_PPGpulse

t = np.linspace(0, 1, 1000)
pulse = gen_PPGpulse(t, pulse_type=1)
```

---

### `gen_PPG_artifacts(duration, prob, dur_mu, RMS_shape, RMS_scale, slope, Fd)`

Generate motion artifacts using Markov chain model.

**Parameters:**
- `duration` (int): Signal duration in samples
- `prob` (array): Probability of each artifact type [4 elements]
- `dur_mu` (array): Mean duration for each artifact [5 elements]
- `RMS_shape`, `RMS_scale`: Gamma distribution parameters
- `slope` (array): PSD slope for each artifact type [4 elements]
- `Fd` (int): Sampling frequency

**Example:**
```python
from ppg_artifacts import gen_PPG_artifacts
from data_loader import load_artifact_params

params = load_artifact_params('data/artifact_param.mat')
prob = np.array([0.25, 0.25, 0.25, 0.25])
dur_mu = np.array([10, 6, 6, 6, 6]) * 1000

artifact = gen_PPG_artifacts(
    duration=10000,
    prob=prob,
    dur_mu=dur_mu,
    RMS_shape=params['RMS_shape'],
    RMS_scale=params['RMS_scale'],
    slope=params['slope_m'],
    Fd=1000
)
```

---

## 🎯 Usage Examples

### Example 1: Different Heart Rates

```python
# Slow heart rate (60 bpm)
RR_slow = np.ones(20) * 1000
PPG_slow, _, _ = gen_PPG(RR_slow, pulse_type=1, Fd=1000)

# Fast heart rate (90 bpm)
RR_fast = np.ones(20) * 667
PPG_fast, _, _ = gen_PPG(RR_fast, pulse_type=1, Fd=1000)
```

**Note**: Respiratory frequency automatically adjusts based on heart rate (HR:Resp ≈ 4.5:1)

### Example 2: HR-Respiratory Coupling Demo

```bash
python examples/demo_hr_resp_coupling.py
```

Demonstrates:
- How respiratory rate changes with heart rate
- FFT spectrum showing HR and respiratory peaks
- Physiologically realistic 4.5:1 ratio

### Example 3: Different Pulse Types

```python
for pulse_type in range(1, 6):
    RR = np.ones(10) * 800
    PPG, _, _ = gen_PPG(RR, pulse_type=pulse_type, Fd=1000)
    # Each type has distinct morphology
```

---

## 🔧 Parameters Guide

### Pulse Types (1-5)

| Type | Description | Clinical Context |
|------|-------------|------------------|
| 1 | Standard morphology | Normal healthy subject |
| 2 | Variant morphology | Common variation |
| 3 | Alternative shape | Age/physiology dependent |
| 4 | Modified waveform | Different measurement sites |
| 5 | Distinctive pattern | Specific conditions |

### Artifact Types

| Index | Type | Description |
|-------|------|-------------|
| 0 | Device displacement | Sensor movement |
| 1 | Forearm motion | Arm movement |
| 2 | Hand motion | Hand/finger movement |
| 3 | Poor contact | Weak sensor coupling |

**Usage:**
```python
# Enable specific artifacts
typ_artifact = np.array([0, 1, 0, 0])  # Device displacement + forearm motion
```

### Rhythm Types

- **SR** (Sinus Rhythm): Normal regular heartbeat
- **AF** (Atrial Fibrillation): Irregular RR intervals

---

## 🧪 Training Data Generation

### Batch Generation

Use `generate_training_data.py` for creating ML datasets:

```bash
python generate_training_data.py
```

**Features:**
- Balanced dataset across pulse types
- Multiple artifact configurations
- Automatic FFT feature extraction
- Labeled data for classification

### Data Structure

Generated `.npz` files contain:
```python
data = np.load('output/sample.npz')

data['PPG']          # Raw PPG signal
data['PPGmodel']     # Pulsatile component
data['artifact']     # Artifact component
data['rr']           # RR intervals
data['PPGpeakIdx']   # Peak indices
data['pulse_type']   # Pulse morphology label
data['rhythm']       # SR or AF
```

---

## 🔬 Physiological Realism Details

### Beat-to-Beat Continuity

Each pulse inherits the baseline from the previous beat:
- **First beat**: Self-continuous (start = end)
- **Subsequent beats**: Start value = previous beat's end value
- **Result**: Zero discontinuity jumps, smooth transitions

### Heart Rate-Respiratory Coupling

Physiological ratio: **HR:Resp ≈ 4.5:1**

Examples:
- HR = 60 bpm → Resp = 13.3 breaths/min (0.22 Hz)
- HR = 75 bpm → Resp = 16.7 breaths/min (0.28 Hz)
- HR = 90 bpm → Resp = 20 breaths/min (0.33 Hz)

### Multi-Source Noise

1. **High-frequency noise** (~1%): Sensor noise, muscle tremor
2. **Low-frequency drift** (~2%): Baseline wander at 0.05 Hz
3. **HRV modulation** (~0.5%): Spectral broadening
4. **Respiratory harmonics**: 2nd harmonic at 15% amplitude

**Result**: Realistic FFT spectrum with broad peaks instead of sharp delta functions.

---

## 📁 Project Structure

```
PPG_generation/
├── README.md                    # This file
├── LICENSE                      # License information
├── requirements.txt             # Python dependencies
│
├── Core Modules
│   ├── ppg_pulse.py            # Pulse waveform generation
│   ├── ppg_generator.py        # Main PPG signal generator
│   ├── ppg_artifacts.py        # Motion artifact generation
│   └── data_loader.py          # Data loading utilities
│
├── Scripts
│   ├── main_ppg.py             # Main demo script
│   └── generate_training_data.py  # Batch data generation
│
├── Examples
│   ├── demo_hr_resp_coupling.py   # HR-respiratory demo
│   └── validate_ppg.py            # Validation script
│
├── Data
│   ├── artifact_param.mat      # Artifact parameters
│   └── pulse_templates.mat     # Pulse template parameters
│
└── output/                     # Generated outputs (created at runtime)
```

---

## 🛠️ Utility Scripts

### Artifact Parameter Verification
```bash
python verify_artifact_params.py
```
Shows detailed explanations of:
- `typ_artifact` probability distributions
- `dur_mu0` and `dur_mu` effects on artifact density
- Artifact percentage calculations
- Configuration examples

### HR-Respiratory Coupling Demo
```bash
python examples/demo_hr_resp_coupling.py
```
Demonstrates how respiratory rate automatically adjusts with heart rate (4.5:1 ratio).

### PPG Validation
```bash
python examples/validate_ppg.py
```
Compares Python-generated PPG with MATLAB templates.

---

## 💾 Data Files

### Included Files

Small parameter files are included in the package:
- `artifact_param.mat` (475 bytes) - Artifact generation parameters
- `pulse_templates.mat` (38 KB) - Pulse morphology parameters

### Large Data Files (Optional Download)

> [!NOTE]
> **Large data files (>100MB) are not included in the Git repository.**
> 
> **Download from Google Drive**: [PPG Large Data Files](https://drive.google.com/drive/folders/15BcK82XtAM-Ggcagsd12yr2iVZHEj6nH?usp=share_link)

**Large data files** (optional, total ~2.4 GB):
- `PPG_1.mat` - Large PPG dataset 1
- `PPG_2.mat` - Large PPG dataset 2  
- `PPG_3.mat` - Large PPG dataset 3
- `DATA_RR_SR_real.mat` (1.7 MB) - Real sinus rhythm RR intervals
- `DATA_RR_AF_real.mat` (7.5 MB) - Real AF RR intervals
- `DATA_PQRST_real.mat` (924 MB) - Real ECG waveforms
- `DATA_f_waves_real.mat` (648 MB) - F-wave data
- `DATA_noises_real.mat` (830 MB) - Noise database

**Installation**:
1. Download files from the Google Drive link above
2. Place them in the `data/` directory
3. The code will automatically detect and use them

**Note**: Code works without large data files by generating synthetic data automatically.

---

## 📊 Output Files

Running `main_ppg.py` generates:

### Visualizations
- `*_PPG.png` - PPG signal with pulsatile and artifact components
- `*_RR.png` - RR interval distribution
- `*_FFT.png` - Frequency spectrum analysis

### Data
- `*.npz` - NumPy compressed format (recommended)
  - Contains: PPG, PPGmodel, artifact, RR, peaks, metadata
- `*_PPG.csv` - PPG signal CSV (optional)
- `*_RR.csv` - RR intervals CSV (optional)

### Loading Saved Data

```python
import numpy as np

data = np.load('output/python_ppg_test.npz')
PPG = data['PPG']
PPGmodel = data['PPGmodel']
artifact = data['artifact']
rr = data['rr']
Fd = data['sampling_freq']
```

---

## 🎓 Citation

If you use this PPG generator in your research, please cite the original papers:

1. **PPG Generation**:
   ```
   Solosenko, A., Petrenas, A., Marozas, V., & Sornmo, L. (2017).
   Modeling of the photoplethysmogram during atrial fibrillation.
   Computers in Biology and Medicine, 81, 130–138.
   ```

2. **Artifact Generation**:
   ```
   Paliakaite, B., Petrenas, A., Solosenko, A., & Marozas, V. (2021).
   Modeling of artifacts in the wrist photoplethysmogram: Application to
   the detection of life-threatening arrhythmias.
   Biomedical Signal Processing and Control, 66, 102421.
   ```

---

## 📜 License

- **PPG Generator Core**: Freeware for private, non-commercial use
- **Artifact Generation**: GNU General Public License v3
- **Python Implementation**: 2024

See [LICENSE](LICENSE) for full details.

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure you're in the correct directory
cd /path/to/PPG_generation

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/PPG_generation"
```

### Missing artifact_param.mat
Code will use default parameters if the file is not found. No action needed.

### Memory Issues
For very long signals, reduce `num_beats` or generate in batches.

---

## 🔄 Version History

### v2.2 (2025-12-18)
- ✅ **Multi-Task UNet Architecture**: Simultaneous Waveform Classification and Noise Segmentation.
- ✅ **Robust Data Generation**: 
  - Chunked saving to prevent memory overflow (50k+ samples).
  - Mixed noise injection (0-20dB SNR).
  - Precise sample-level artifact masks.
- ✅ **Remote Training Support**:
  - SLURM scripts for independent Data Generation and Training.
  - Deployment-ready `.tar.gz` packaging logic.
- ✅ **Real-Time Inference Ready**:
  - `inference.py` for live signal processing (requires trained model).

### v1.1 (2024-12-17)
- ✅ **FFT Peak Annotations**: Automatic detection and labeling of top 4 frequency peaks
- ✅ **Noise Reduction**: Reduced to 0.1% for clean waveforms matching MATLAB reference
- ✅ **Artifact Parameter Guide**: Added `verify_artifact_params.py` for parameter tuning
- ✅ **Enhanced Documentation**: Detailed artifact control and noise level explanations
- ✅ **Improved Visualizations**: Color-coded peak markers with frequency and bpm labels

### v1.0 (2024-12-17)
- ✅ Perfect beat-to-beat continuity with baseline inheritance
- ✅ HR-respiratory coupling (physiological 4.5:1 ratio)
- ✅ Multi-source physiological noise for realistic FFT
- ✅ Cubic spline baseline correction
- ✅ MATLAB-calibrated pulse parameters
- ✅ Comprehensive documentation

---

## 🧠 V2.0 Deep Learning Workflow

### 1. Data Generation (Segmentation)
Generates `(Signal, Mask, Label)` triplets for UNet training.
```bash
# Generate 50,000 samples with chunking
python generate_segmentation_data.py --num_samples 50000 --output_dir ml_dataset_v2
```

### 2. Training (Multi-Task UNet)
Trains the model to predict both **Waveform Class (1-5)** and **Artifact Mask (0-4)**.
```bash
# Train on GPU
python train_segmentation.py --data_path ml_dataset_v2
```

### 3. Remote Deployment
Use the included `deploy_and_setup.sh` or SLURM scripts (`slurm_datagen_v2.sh`, `slurm_train_v2.sh`) for cluster execution.

---

## 👥 Credits

**Original MATLAB Implementation:**
- **Source**: [PhysioNet: ECG and PPG signals simulator for arrhythmia detection (v1.3.1)](https://physionet.org/content/ecg-ppg-simulator-arrhythmia/1.3.1/)
- Andrius Solosenko (PPG generation)
- Birute Paliakaite (Artifact generation)
- Biomedical Engineering Institute, Kaunas University of Technology

**Python Implementation:**
- 2024 conversion with physiological enhancements

---

## 📧 Support

For questions or issues, please refer to:
1. This README documentation
2. Example scripts in `examples/`
3. Code comments in source files

---

**Happy Signal Processing! 🚀**
