#!/usr/bin/env python3
"""
Verify v3.0 CWT Input Tensor
Visualizes the 34-channel input (Amp + Vel + 32 CWT Scales)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Ensure local modules are found
sys.path.append(os.getcwd())

from ml_training.dataset import PPGDataset

def verify_cwt_tensor():
    print("=" * 70)
    print("Verifying v3.0 CWT Input Tensor")
    print("=" * 70)
    
    # Check for generated data
    data_dir = "test_release_dataset" # Should exist from previous steps
    if not os.path.exists(data_dir):
        print(f"[WARN] {data_dir} not found. Running generate_training_data.py for 5 samples...")
        os.system("python3 generate_training_data.py --num_samples 5 --output_dir test_release_dataset")
    
    if not os.path.exists(data_dir):
        print("[FAIL] Could not generate data.")
        return

    # Load Dataset
    print(f"[-] Loading dataset from {data_dir}...")
    dataset = PPGDataset(data_dir, task='waveform', max_length=8000)
    
    if len(dataset) == 0:
        print("[FAIL] Dataset is empty.")
        return
        
    # Get a sample
    print(f"[-] Fetching sample 0...")
    signal_tensor, label = dataset[0] # Returns [34, 8000]
    
    # Check Shape
    print(f"[INFO] Tensor Shape: {signal_tensor.shape}")
    expected_shape = (34, 8000)
    if signal_tensor.shape == expected_shape:
        print("[SUCCESS] Shape matches expectation (34 channels).")
    else:
        print(f"[FAIL] Expected {expected_shape}, got {signal_tensor.shape}")
        return

    # Visualization
    print("[-] Visualizing Input Channels...")
    
    signal_np = signal_tensor.numpy()
    
    # Extract components
    amp = signal_np[0, :]
    vel = signal_np[1, :]
    cwt = signal_np[2:, :] # [32, 8000]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 1. Amplitude
    ax1.plot(amp, 'k-', linewidth=0.8)
    ax1.set_title("Channel 0: Amplitude (Normalized)", fontweight='bold')
    ax1.set_ylabel("Amp")
    ax1.grid(True, alpha=0.3)
    
    # 2. Velocity
    ax2.plot(vel, 'b-', linewidth=0.8)
    ax2.set_title("Channel 1: Velocity (Normalized)", fontweight='bold')
    ax2.set_ylabel("Vel")
    ax2.grid(True, alpha=0.3)
    
    # 3. CWT Spectrogram
    # CWT shape is [32, Length]. Scales 1 (High Freq) to 32 (Low Freq).
    # We want low freq at bottom, so usually we might flip, but here Row 0 is Scale 1 (High).
    # Let's plot as image.
    im = ax3.imshow(cwt, aspect='auto', cmap='jet', interpolation='nearest', 
               extent=[0, 8000, 32, 1], origin='upper')
    ax3.set_title("Channels 2-33: CWT Spectrogram (Scales 1-32)", fontweight='bold')
    ax3.set_ylabel("Scale (1=High Freq, 32=Low Freq)")
    ax3.set_xlabel("Time (Samples)")
    plt.colorbar(im, ax=ax3, label="Energy (Z-Score)")
    
    plt.tight_layout()
    out_file = "verify_cwt_vis.png"
    plt.savefig(out_file, dpi=150)
    print(f"[SUCCESS] Visualization saved to {out_file}")
    print("\nVisual Inspection Guide:")
    print("  - Look at the CWT plot.")
    print("  - Healthy beats should show regular 'blobs' of energy at lower scales (periodic).")
    print("  - Sharp artifacts should show vertical lines spanning many scales.")

if __name__ == "__main__":
    try:
        verify_cwt_tensor()
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        print("Please run: pip install scipy")
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
