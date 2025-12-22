#!/usr/bin/env python3
"""
v4.0 Dual-Stream Model Verification Script
Batch generates signals with different pulse types, noise types, and noise durations.
Outputs images with detailed titles showing GT vs Prediction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ppg_generator import gen_PPG
from ml_training.model_factory import create_model
from ml_training.dataset import preprocess_signal

# --- Inline Preprocessing Removed (Using imported version) ---


# Label mappings
PULSE_LABELS = {0: 'Sinus (N)', 1: 'Premature (S)', 2: 'Ventricular (V)', 3: 'Fusion (F)', 4: 'Unknown (Q)'}
NOISE_LABELS = {0: 'Clean', 1: 'Baseline', 2: 'Forearm', 3: 'Hand', 4: 'HighFreq'}


# --- Artifact Generation (Ported from ppg_artifacts.py) ---
from scipy import signal as scipy_signal

def generate_specific_artifact(duration_samples, artifact_type_idx, fs=1000):
    """Generate specific artifact type using FIR filtering method used in training"""
    # Parameters from ppg_artifacts.py / generate_training_data.py
    # idx 1: Device Disp, 2: Forearm, 3: Hand, 4: Poor Contact
    # Adjusted to match the artifact types in your labels
    
    # Slopes and Gamma params (approximate from original paper/code)
    # Type: [Disp, Forearm, Hand, HighFreq/PoorContact]
    slopes = [-1.8, -2.5, -3.0, -1.0] 
    rms_shapes = [2.0, 2.5, 3.0, 1.5]
    rms_scales = [0.5, 0.6, 0.7, 0.4]
    
    idx = artifact_type_idx - 1 # 0-indexed
    if idx < 0 or idx >= 4: return np.zeros(duration_samples)
    
    slope = slopes[idx]
    shape = rms_shapes[idx]
    scale = rms_scales[idx]
    
    # Design FIR filter
    fv = np.linspace(0, 1, 100)
    a = np.zeros(100)
    a[0:2] = 0
    freq_hz = fv[2:] * (fs / 2)
    a[2:] = slope * np.log10(freq_hz)
    a = np.sqrt(10 ** (a / 10))
    b = scipy_signal.firls(251, fv, a)
    
    # Filter noise
    noise = np.random.randn(duration_samples + 300)
    filtered = scipy_signal.lfilter(b, 1, noise)[300:]
    
    # Normalize and Scale
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-6)
    rms = np.random.gamma(shape, scale)
    
    # Amplitude boost to ensure visibility/challenge
    return filtered * rms * 1.5

def add_artifact(signal, artifact_type, start_ratio=0.3, duration_ratio=0.3, fs=1000):
    """Add artifact to signal at specified position and duration."""
    signal = signal.copy()
    n = len(signal)
    
    if artifact_type == 0:
        return signal, np.zeros(n, dtype=np.int64)
    
    start = int(n * start_ratio)
    duration = int(n * duration_ratio)
    
    # Generate realistic artifact
    artifact_segment = generate_specific_artifact(duration, artifact_type, fs)
    
    # Add to signal
    end = start + len(artifact_segment)
    if end > n:
        end = n
        artifact_segment = artifact_segment[:n-start]
        
    signal[start:end] += artifact_segment
    
    # Create mask
    mask = np.zeros(n, dtype=np.int64)
    mask[start:end] = artifact_type
    
    return signal, mask


def generate_sample(pulse_type, artifact_type, duration_sec=8, fs=1000, noise_duration_ratio=0.3):
    """Generate a single PPG sample with specified pulse and artifact types."""
    # Generate RR intervals
    n_beats = int(duration_sec * 1.2)  # ~72 bpm average
    rr_base = 1000 / 1.2  # ms
    RR = rr_base + np.random.randn(n_beats) * 50
    RR = np.clip(RR, 600, 1200)
    
    # Generate PPG
    ppg, _, _ = gen_PPG(RR.astype(int), pulse_type=pulse_type + 1, Fd=fs)
    
    # Trim/pad to exact length
    target_len = duration_sec * fs
    if len(ppg) > target_len:
        ppg = ppg[:target_len]
    else:
        ppg = np.pad(ppg, (0, target_len - len(ppg)), mode='edge')
    
    # Add artifact
    ppg_noisy, mask = add_artifact(ppg, artifact_type, duration_ratio=noise_duration_ratio)
    
    return ppg_noisy, mask, pulse_type


def run_inference(model, signal, device):
    """Run model inference on a single signal."""
    # Preprocess
    tensor = preprocess_signal(signal)
    tensor = tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_clf, pred_seg = model(tensor)
    
    # Get predictions
    pred_pulse = pred_clf.argmax(1).item()
    pred_mask = pred_seg.argmax(1).squeeze().cpu().numpy()
    
    # Get confidence
    clf_probs = torch.softmax(pred_clf, dim=1)
    clf_conf = clf_probs.max().item()
    
    # Determine dominant noise type from mask
    unique, counts = np.unique(pred_mask, return_counts=True)
    noise_counts = {u: c for u, c in zip(unique, counts) if u != 0}
    if noise_counts:
        pred_noise = max(noise_counts, key=noise_counts.get)
    else:
        pred_noise = 0
    
    return pred_pulse, pred_noise, pred_mask, clf_conf


def verify_batch(model_path, output_dir, device='cpu'):
    """Run batch verification across all pulse types and noise types."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = create_model('dual', in_channels=34, n_classes_seg=5, n_classes_clf=5, attention=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    
    # Test configurations
    pulse_types = [0, 1, 2, 3, 4]  # All 5 pulse types
    noise_types = [0, 1, 2, 3, 4]  # Clean + 4 artifact types
    noise_durations = [0.1, 0.3, 0.5, 0.7]   # 10%, 30%, 50%, 70% of signal
    
    results = []
    
    for pt in pulse_types:
        for nt in noise_types:
            for nd in noise_durations:
                # Generate sample
                signal, gt_mask, gt_pulse = generate_sample(pt, nt, noise_duration_ratio=nd)
                
                # Run inference
                pred_pulse, pred_noise, pred_mask, conf = run_inference(model, signal, device)
                
                # Record result
                pulse_correct = pred_pulse == gt_pulse
                noise_correct = pred_noise == nt
                results.append({
                    'gt_pulse': pt, 'gt_noise': nt, 'duration': nd,
                    'pred_pulse': pred_pulse, 'pred_noise': pred_noise,
                    'pulse_correct': pulse_correct, 'noise_correct': noise_correct
                })
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(14, 5))
                
                # Plot Signal
                t = np.arange(len(signal)) / 1000
                ax.plot(t, signal, 'k-', linewidth=0.8, alpha=0.8, label='Signal')
                
                # Overlay Ground Truth Noise (Green Background)
                # Find contiguous segments for GT
                gt_mask_bool = (gt_mask > 0)
                gt_segments = np.ma.clump_masked(np.ma.masked_array(t, mask=~gt_mask_bool))
                for s in gt_segments:
                    ax.axvspan(t[s.start], t[s.stop-1], color='green', alpha=0.2, label='GT Noise' if 'GT Noise' not in [l.get_label() for l in ax.get_lines() + ax.patches] else "")

                # Overlay Predicted Noise (Red Background/Bottom Bar)
                # Using bottom bar to avoid color blending confusion if they overlap perfectly
                # Or use different hatching?
                # Let's use Red highlight but slightly offset or hatching.
                # Actually, user asked for "mark ... directly on row data".
                # Standard practice: GT = Green Background, Pred = Red Hatching or Just Red Background (Mix = Yellow/Brown)
                
                pred_mask_bool = (pred_mask > 0)
                pred_segments = np.ma.clump_masked(np.ma.masked_array(t, mask=~pred_mask_bool))
                for s in pred_segments:
                    # Use a hatched pattern or border for prediction to distinguish from solid GT
                    ax.axvspan(t[s.start], t[s.stop-1], facecolor='none', edgecolor='red', hatch='//', alpha=0.5, linewidth=0, label='Pred Noise' if 'Pred Noise' not in [l.get_label() for l in ax.get_lines() + ax.patches] else "")
                    ax.axvspan(t[s.start], t[s.stop-1], color='red', alpha=0.1) # Light red tint
                
                ax.set_ylabel('Amplitude')
                ax.set_xlabel('Time (s)')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
                
                # Update Labels with Type X
                pulse_label_expanded = f"Type {pt+1}: {PULSE_LABELS[pt]}"
                
                # Title with GT vs Prediction
                gt_str = f"GT: {pulse_label_expanded} + {NOISE_LABELS[nt]}"
                pred_str = f"Pred: {PULSE_LABELS[pred_pulse]} ({conf*100:.1f}%) + {NOISE_LABELS[pred_noise]}"
                
                # Color code correctness
                pulse_color = '✓' if pulse_correct else '✗'
                noise_color = '✓' if noise_correct else '✗'
                
                fig.suptitle(f"{gt_str}  |  {pred_str}  [Pulse:{pulse_color} Noise:{noise_color}]", 
                            fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                
                # Save figure
                fname = f"v4_Type{pt+1}_{NOISE_LABELS[nt]}_N{nt}_D{int(nd*100)}.png"
                plt.savefig(os.path.join(output_dir, fname), dpi=150)
                plt.close()
                
                print(f"  Saved: {fname} | Pulse: {pulse_color} | Noise: {noise_color}")
    
    # Summary
    pulse_acc = sum(r['pulse_correct'] for r in results) / len(results)
    noise_acc = sum(r['noise_correct'] for r in results) / len(results)
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Total Samples: {len(results)}")
    print(f"Pulse Classification Accuracy: {pulse_acc*100:.1f}%")
    print(f"Noise Type Accuracy: {noise_acc*100:.1f}%")
    print(f"Results saved to: {output_dir}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='v4.0 Dual-Stream Verification')
    parser.add_argument('model_path', type=str, nargs='?', default='output/v4_best_model.pth', help='Path to best_model.pth')
    parser.add_argument('--output_dir', type=str, default='validation_v4', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    verify_batch(args.model_path, args.output_dir, args.device)
