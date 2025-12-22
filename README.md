# PPG Signal Generator with Dual-Stream ML Architecture

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Model Accuracy](https://img.shields.io/badge/Classification-100%25-green)](https://github.com/Pasdeau/PPG_Generator)
[![Segmentation](https://img.shields.io/badge/Segmentation-99.7%25-green)](https://github.com/Pasdeau/PPG_Generator)

A comprehensive Python toolkit for **physiologically-accurate PPG signal generation** and **real-time artifact detection** using deep learning.

## 🎯 Key Features

- **Synthetic PPG Generation**: Realistic simulation of 5 cardiac arrhythmia types with 4 motion artifact categories
- **v4.0 Dual-Stream Model**: State-of-the-art deep learning achieving:
  - ✅ **100% Classification Accuracy** (Sinus Rhythm, PVC, PAC, Fusion, Unknown)
  - ✅ **99.7% Segmentation Accuracy** (Pixel-level artifact localization)
- **Real-time Streaming Inference**: <75ms latency for continuous ADS1298 integration
- **Hardware Integration**: Serial receiver for ADS1298 8-channel PPG acquisition

---

## 📖 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Pasdeau/PPG_Generator.git
cd PPG_Generator

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Your First PPG Signal

```bash
# Generate 10-beat PPG with artifacts
python main_ppg.py --num_beats 10 --pulse_type 1 --artifact_type 2

# Output files:
# - output/python_ppg_test_PPG.png   # Time-domain plot
# - output/python_ppg_test_FFT.png   # Frequency analysis
# - output/python_ppg_test.npz       # Raw data
```

### Use Pre-trained v4.0 Model

```python
from ppg_generator import gen_PPG
from ml_training.model_factory import load_model
import numpy as np

# Generate test signal
RR = np.array([800] * 10)  # 10 beats, 800ms RR interval
ppg, peaks, _ = gen_PPG(RR, pulse_type=1, Fd=1000)

# Load v4.0 model
model = load_model('output/v4_best_model.pth')

# Predict
pulse_class, artifact_mask = model.predict(ppg)
print(f"Detected: {pulse_class}, Artifact coverage: {artifact_mask.mean():.1%}")
```

---

## 🏗️ Architecture Overview

### v4.0 Dual-Stream Design

Our model solves the **Task Conflict** problem in multi-task learning by using two independent processing streams:

```
Input Signal (8s @ 1000Hz)
    ├─→ Stream A (ResNet1D) ────→ Pulse Classification (5 classes)
    │   Input: [2-ch Time-Domain]
    │
    └─→ Stream B (SE-UNet) ─────→ Artifact Segmentation (5 classes/pixel)
        Input: [34-ch CWT Tensor]
```

**Key Innovation**: Decoupled feature extraction eliminates gradient conflict between classification (needs translation-invariance) and segmentation (needs translation-equivariance).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details.

---

## 📊 Performance Benchmarks

| Model | Architecture | Clf Acc (%) | Seg Acc (%) | Params |
|-------|-------------|------------|------------|--------|
| v2.4  | ResNet1D    | 98.5       | --         | 1.2M   |
| v3.0  | UNet+CWT    | 19.5       | 98.5       | 8.4M   |
| v3.1  | UNet+Attn   | 99.9       | 99.2       | 9.1M   |
| **v4.0** | **Dual-Stream** | **100.0** | **99.69** | **10.8M** |

*Tested on 20,000 synthetic samples with balanced artifact distribution*

---

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Technical deep-dive into v4.0 Dual-Stream model
- **[DATASET.md](docs/DATASET.md)**: How to generate custom training datasets
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)**: Real-time inference setup guide
- **[API.md](docs/API.md)**: Function reference and examples

---

## 🔬 Project Structure

```
PPG_Generator/
├── ppg_generator.py           # Core PPG synthesis engine
├── ppg_artifacts.py            # Motion artifact simulation
├── ppg_pulse.py                # Pulse template library
├── main_ppg.py                 # CLI demo
│
├── ml_training/                # Deep learning pipeline
│   ├── models/                 # ResNet1D, UNet, Dual-Stream
│   ├── dataset.py              # CWT preprocessing
│   └── train_dual_stream.py    # v4.0 training script
│
├── generate_training_data.py   # Dataset generation
├── verify_dual_v4.py           # Model validation
├── streaming_inference_demo.py # Real-time inference test
│
├── ADS1298_Receiver_v4/        # Hardware integration
│   ├── ads1298_serial.py       # Serial port receiver
│   └── v4_best_model.pth       # Pre-trained weights
│
├── examples/                   # Usage examples
├── output/                     # Generated signals & models
└── archive/                    # Legacy code (v1.x-v3.x)
```

---

## 🚀 Advanced Usage

### Generate Custom Training Dataset

```bash
# Create 20,000 samples with balanced classes
python generate_training_data.py --num_samples 20000 --output data/
```

### Train from Scratch

```bash
# Requires GPU (CUDA)
python ml_training/train_dual_stream.py \
    --data data/training_set.npz \
    --epochs 50 \
    --batch_size 32
```

### Real-time Streaming (ADS1298 Integration)

```bash
# Test streaming inference capability
python streaming_inference_demo.py

# Expected latency: <75ms per 8-second window
# Throughput: >13kHz (far exceeds 1kHz sampling rate)
```

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2024ppg,
  title={Dual-Stream Attention Network for Simultaneous Noise Segmentation and Waveform Classification in PPG},
  author={Wang, Wenzheng},
  journal={[Under Review]},
  year={2024}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for improvement**:
- Clinical dataset validation
- Additional cardiac arrhythmia types
- Model compression for edge deployment
- Multi-modal fusion (PPG + ECG + Accelerometer)

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

Key points:
- ✅ Free to use, modify, and distribute
- ✅ Must disclose source if distributed
- ❌ Cannot be used in proprietary software without GPL compliance

---

## 🙏 Acknowledgments

- Original MATLAB PPG simulator: [Birute Paliakaite et al., 2021](https://doi.org/10.1016/j.bspc.2021.102421)
- SE-Block: [Hu et al., CVPR 2018](https://arxiv.org/abs/1709.01507)
- Continuous Wavelet Transform: PyWavelets library

---

## 📞 Contact

- **Author**: Wenzheng Wang
- **Email**: wenzheng.wang@lip6.fr
- **Institution**: Sorbonne Université, CNRS, LIP6, Paris, France
- **GitHub**: [@Pasdeau](https://github.com/Pasdeau)

---

**⭐ Star this repo if you find it useful!**
