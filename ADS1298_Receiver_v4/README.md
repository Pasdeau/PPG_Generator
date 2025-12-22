# ADS1298 Intelligent Receiver (v4.0)

This folder contains the complete, self-contained software for the ADS1298 PPG Receiver, integrated with the v4.0 Dual-Stream AI Model for real-time Clean/Noise diagnosis.

## 📦 Contents

- `ads1298_serial.py`: Main application code (PyQt6 + PyTorch).
- `v4_best_model.pth`: The pre-trained Dual-Stream AI model (100% Classification Accuracy).
- `ml_training/`: Required support library for model architecture and data preprocessing.

## 🚀 Setup & Run

### 1. Requirements

You need a Python environment with PyTorch and PyQt6/PyQtGraph.

```bash
pip install torch numpy scipy matplotlib pyqt6 pyqtgraph pyserial
```

### 2. Running the Receiver

Simply run the script. It will automatically detect the model in the same folder.

```bash
python ads1298_serial.py
```

### 3. Features

- **Real-Time Plotting**: Visualizes CH5/CH6 (Raw) and CH7 (LED Slices).
- **AI Diagnostics**:
  - Automatically analyzes **CH7** (LED Slices).
  - Initializes 4 dedicated detectors (LED1-LED4) sharing one efficient model.
  - Outputs "Clean" or specific Noise Type in the console (e.g., `[AI] LED2: Sinus (N) | Noise: Forearm (15%)`).

## ⚠️ Portability Note

To move this receiver to another computer, simply **copy this entire folder**. 

**Do not separate** `v4_best_model.pth` or the `ml_training` folder from `ads1298_serial.py`, as they are required for the AI features to work.
