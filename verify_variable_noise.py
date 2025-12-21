"""
Verify Robustness to Variable Noise Lengths
v3.1 SE-Attention Model

This script manually constructs signals with specific noise durations to test
segmentation precision.

Author: Wenzheng Wang
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from ppg_pulse import gen_PPGpulse
# from ppg_artifacts import add_forearm_motion # Not available directly
from ml_training.dataset import cwt_ricker
from ml_training.models.unet_ppg import UNetPPG
from collections import OrderedDict

def add_forearm_motion(signal, fs):
    """Add forearm motion artifact (mid-frequency oscillation)."""
    t = np.linspace(0, len(signal)/fs, len(signal))
    noise = 0.25 * np.sin(2 * np.pi * 2.5 * t) + 0.15 * np.sin(2 * np.pi * 4 * t)
    return signal + noise + 0.1 * np.random.randn(len(signal))

def load_v3_1_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetPPG(in_channels=34, n_classes_seg=5, n_classes_clf=5, attention=True).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle DataParallel prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model, device

def create_input_tensor(signal):
    # Z-norm
    u = np.mean(signal)
    s = np.std(signal) + 1e-6
    sig_norm = (signal - u) / s
    
    # Velocity
    vel = np.diff(signal, prepend=signal[0])
    u_v = np.mean(vel)
    s_v = np.std(vel) + 1e-6
    vel_norm = (vel - u_v) / s_v
    
    # CWT
    scales = np.arange(1, 33)
    cwt = cwt_ricker(signal, scales)
    # Norm CWT
    cwt = (cwt - np.mean(cwt)) / (np.std(cwt) + 1e-6)
    
    # Stack: [Amplitude, Velocity, CWT x 32]
    input_tensor = np.vstack([sig_norm, vel_norm, cwt])
    return torch.FloatTensor(input_tensor).unsqueeze(0)  # [1, 34, 8000]

def test_variable_noise(model, device, output_dir):
    fs = 1000
    L = 8000
    t = np.linspace(0, 8, L)
    
    # 1. Generate Clean Pulse Train
    clean_sig = np.zeros(L)
    beat_interval = 800 # 0.8s
    for loc in range(0, L, beat_interval):
        pulse = gen_PPGpulse(np.linspace(0, 1, beat_interval), pulse_type=1)
        end = min(loc + len(pulse), L)
        clean_sig[loc:end] = pulse[:end-loc]
        
    # 2. Inject Variable Noise
    # Case A: Short Burst (0.5s) at 1s
    # Case B: Medium Burst (1.5s) at 3s
    # Case C: Long Burst (2.5s) at 5.5s
    
    noisy_sig = clean_sig.copy()
    mask_gt = np.zeros(L)
    
    noise_defs = [
        (1000, 1500, "Short (0.5s)"),
        (3000, 4500, "Medium (1.5s)"),
        (5500, 8000, "Long (2.5s)")
    ]
    
    # Generate long noise to slice from
    long_noise = add_forearm_motion(np.zeros(L), fs)
    
    for start, end, label in noise_defs:
        # Inject noise
        # Scale noise to specific SNR-ish level
        segment_len = end - start
        noise_segment = long_noise[start:end] 
        # Normalize roughly
        noise_segment = noise_segment * 0.5 
        noisy_sig[start:end] += noise_segment
        mask_gt[start:end] = 2 # Class 2 = Forearm Motion
        
    # 3. Inference
    x = create_input_tensor(noisy_sig).to(device)
    with torch.no_grad():
        out_clf, out_mask = model(x)
        pred_mask = torch.argmax(out_mask, dim=1).cpu().numpy()[0]
        prob_clf = torch.softmax(out_clf, dim=1).cpu().numpy()[0]
        
    # 4. Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Ax 1: Signal
    axes[0].plot(t, noisy_sig, 'k-', linewidth=0.8, label='Noisy Input')
    axes[0].set_title('Input Signal with Variable Length Artifacts', fontweight='bold')
    axes[0].set_ylabel('Amp')
    
    # Ax 2: GT Mask
    axes[1].plot(t, mask_gt, 'g-', linewidth=2, label='Ground Truth')
    axes[1].fill_between(t, 0, mask_gt, color='green', alpha=0.3)
    axes[1].set_title('Ground Truth Segmentation', fontweight='bold')
    axes[1].set_ylabel('Class')
    axes[1].set_ylim([-0.5, 4.5])
    
    # Ax 3: Pred Mask
    axes[2].plot(t, pred_mask, 'r--', linewidth=2, label='Prediction (v3.1)')
    axes[2].fill_between(t, 0, pred_mask, color='red', alpha=0.3)
    axes[2].set_title(f'v3.1 Prediction (Pulse Class: {np.argmax(prob_clf)}={prob_clf[np.argmax(prob_clf)]*100:.1f}%)', fontweight='bold')
    axes[2].set_ylabel('Class')
    axes[2].set_ylim([-0.5, 4.5])
    axes[2].set_xlabel('Time (s)')
    
    # Annotate durations
    for start, end, label in noise_defs:
        axes[0].axvspan(start/1000, end/1000, color='yellow', alpha=0.1)
        axes[0].text((start+end)/2000, max(noisy_sig), label, ha='center', fontsize=9, backgroundcolor='white')
        
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'variable_noise_test.png')
    plt.savefig(save_path, dpi=300)
    print(f"[+] Saved {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 verify_variable_noise.py <model_path>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    model, device = load_v3_1_model(model_path)
    test_variable_noise(model, device, 'validation_special')
