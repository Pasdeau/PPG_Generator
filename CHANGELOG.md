# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [4.0.0] - 2024-12-22

### 🎉 Major Release: Dual-Stream Architecture

#### Added
- **Dual-Stream Model Architecture**: Separate ResNet1D (classification) and SE-UNet (segmentation) branches
  - Resolves "Task Conflict" between translation-invariant and translation-equivariant features
  - Achieves **100% classification accuracy** (up from 99.9%)
  - Achieves **99.69% segmentation accuracy** (up from 99.2%)
- **Real-time Streaming Inference Support**:
  - `streaming_inference_demo.py`: Simulates continuous ADS1298 data reception
  - Average latency: 75ms per 8-second window
  - Throughput: >13kHz (far exceeds 1000Hz sampling rate)
- **ADS1298 Hardware Integration**:
  - Serial receiver with v4.0 model integration
  - Sliding window buffer for continuous processing
- **Documentation**:
  - Comprehensive README with badges and quick start
  - CHANGELOG (this file)
  - CONTRIBUTING guide
  - Streaming capability technical report

#### Changed
- **Model Architecture**: Replaced single-encoder MTL with decoupled dual-stream design
- **Input Processing**: 
  - Stream A: 2-channel time-domain (amplitude + velocity)
  - Stream B: 34-channel CWT tensor (time-frequency representation)
- **Training Script**: `train_dual_stream.py` with joint loss optimization
- **Verification**: Enhanced `verify_dual_v4.py` with overlay visualization

#### Deprecated
- v3.1 single-stream SE-UNet model (moved to `archive/`)
- v3.0 UNet+CWT model (moved to `archive/`)

#### Removed
- Obsolete verification scripts for v2.x models

---

## [3.1.0] - 2024-12-19

### Added
- **SE-Attention Mechanism**: Squeeze-and-Excitation blocks in UNet
- **Balanced Multi-Task Loss**: Equal weighting for classification and segmentation
- **CWT Input Tensor**: 34-channel time-frequency representation

### Changed
- Classification accuracy: 19.5% → 99.9% (fixed gradient dominance issue)
- Segmentation accuracy: 98.5% → 99.2%

---

## [3.0.0] - 2024-12-18

### Added
- **Continuous Wavelet Transform (CWT)**: Time-frequency domain analysis
- **Multi-Task Learning**: Joint classification and segmentation
- 34-channel input tensor (raw + velocity + 32 CWT scales)

### Changed
- Switched from time-domain to time-frequency input

### Known Issues
- Task Conflict: Classification severely degraded (19.5% accuracy)
- Gradient dominance: Segmentation task overwhelms classification

---

## [2.4.0] - 2024-12-17

### Added
- **Dual-channel input**: Amplitude + First derivative (velocity)
- Data augmentation: Variable noise intervals
- Remote GPU training pipeline (SLURM)

### Changed
- Classification accuracy: 95% → 98.5%
- Model: ResNet1D with 4 residual blocks

---

## [2.0.0] - 2024-12-16

### Added
- Initial ResNet1D model for pulse classification
- 5-class classification: SR, PAC, PVC, Fusion, Unknown
- Python PPG generator (ported from MATLAB)
- 4 artifact types: BW, Forearm Motion, Hand Motion, EMG

---

## [1.0.0] - 2024-12-15

### Initial Release
- Fork from original MATLAB PPG simulator
- Basic signal generation without ML

---

## Unreleased / Roadmap

### Planned for v5.0
- [ ] Clinical dataset validation (real PPG from wearables)
- [ ] Model compression (quantization + pruning for edge deployment)
- [ ] Multi-modal fusion (PPG + ECG + Accelerometer)
- [ ] Additional arrhythmia types (Atrial Fibrillation, Tachycardia)
- [ ] ONNX export for cross-platform inference

### Under Consideration
- [ ] Real-time GUI for ADS1298 receiver
- [ ] Docker container for reproducibility
- [ ] ReadTheDocs integration
- [ ] CI/CD with automated testing

---

## Version History Summary

| Version | Date       | Key Achievement                  |
|---------|------------|-----------------------------------|
| v4.0    | 2024-12-22 | 100% Clf, Dual-Stream Architecture |
| v3.1    | 2024-12-19 | 99.9% Clf, SE-Attention           |
| v3.0    | 2024-12-18 | CWT Input, MTL Baseline           |
| v2.4    | 2024-12-17 | 98.5% Clf, Dual-channel Input     |
| v2.0    | 2024-12-16 | Initial ResNet1D                  |
| v1.0    | 2024-12-15 | Signal Generation Only            |

---

**How to Reference a Specific Version:**
```bash
git checkout v4.0.0
```

See [Releases](https://github.com/Pasdeau/PPG_Generator/releases) for downloadable archives.
