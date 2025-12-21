#!/usr/bin/env python3
"""
V2.0 Data Generator: Segmentation & Robustness
Generates (Signal, Mask, Label) triplets for UNet training.
"""

import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from ppg_generator import gen_PPG
from data_loader import generate_synthetic_rr
from ppg_artifacts import gen_PPG_artifacts

def generate_segmentation_sample(sample_id, Fd=1000):
    """
    Generates a single sample with precise segmentation mask.
    
    Returns:
    - signal: (8000,) normalized signal with artifacts
    - mask: (8000,) integer mask (0=clean, 1-4=artifact types)
    - label: int (pulse type 1-5)
    """
    # 1. Base Parameters
    pulse_type = np.random.randint(1, 6)
    rhythm_type = 'SR' if np.random.random() > 0.2 else 'AF'
    
    # 2. Generate Clean Signal
    RR = generate_synthetic_rr(num_beats=60, rhythm_type=rhythm_type)
    clean_signal, _, _ = gen_PPG(RR, pulse_type=pulse_type, Fd=Fd)
    
    # Normalize clean signal
    clean_signal = (clean_signal - np.mean(clean_signal)) / np.std(clean_signal)
    
    # Ensure length 8000
    target_len = 8000
    if len(clean_signal) < target_len:
        clean_signal = np.pad(clean_signal, (0, target_len - len(clean_signal)))
    else:
        clean_signal = clean_signal[:target_len]

    # 3. Generate Artifact Mask & Signal
    # We want precise control, so we'll generate artifacts separately and overlay them
    mask = np.zeros(target_len, dtype=int)
    final_signal = clean_signal.copy()
    
    # Decide if we add artifacts (80% chance)
    if np.random.random() < 0.8:
        # Determine number of artifact segments (1 to 3)
        num_segments = np.random.randint(1, 4)
        
        for _ in range(num_segments):
            # Random artifact type (1-4)
            # 1: Device Disp, 2: Forearm combined, 3: Hand, 4: Poor Contact
            art_type = np.random.randint(1, 5)
            
            # Random duration (0.5s to 3s)
            dur_samples = int(np.random.uniform(0.5, 3.0) * Fd)
            
            # Random start position
            if target_len - dur_samples <= 0: continue
            start_idx = np.random.randint(0, target_len - dur_samples)
            end_idx = start_idx + dur_samples
            
            # Generate artifact snippet (simple Gaussian noise colored by filter)
            # For V2 simplified generation, use white noise with varying amplitude standard deviation
            # Real ppg_artifacts.py is complex, here we approximate for speed/mask precision
            noise = np.random.randn(dur_samples)
            
            # Type-specific modulation
            if art_type == 1: # High amplitude baseline shift
                 noise = np.cumsum(noise) * 0.1 
            elif art_type == 4: # High freq noise
                 noise = noise * 2.0
            
            # Normalize noise snippet to have significant power
            noise_amp = np.random.uniform(0.5, 2.0) * np.std(clean_signal[start_idx:end_idx])
            if np.std(noise) > 0:
                noise = (noise - np.mean(noise)) / np.std(noise) * noise_amp
            
            # Add to signal
            final_signal[start_idx:end_idx] += noise
            
            # Update Mask (overwrite previous)
            mask[start_idx:end_idx] = art_type

    # 4. Add Global White Noise (Robustness)
    # SNR from 0dB to 20dB
    snr = np.random.uniform(0, 20)
    noise_power = np.var(final_signal) / (10 ** (snr / 10))
    global_noise = np.random.randn(target_len) * np.sqrt(noise_power)
    
    final_signal = final_signal + global_noise
    
    # Final Normalization
    final_signal = (final_signal - np.mean(final_signal)) / (np.std(final_signal) + 1e-8)
    
    return final_signal, mask, pulse_type - 1 # 0-indexed label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--output_dir', type=str, default='dataset_v2')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Pre-allocate large arrays for efficiency
    # Signals: (N, 1, 8000) float32
    # Masks: (N, 8000) uint8
    # Labels: (N,) int64
    
    signals = []
    masks = []
    labels = []
    
    print(f"Generating {args.num_samples} samples for V2 Segmentation...")
    
    chunk_size = 1000
    chunk_idx = 0
    
    current_signals = []
    current_masks = []
    current_labels = []
    
    for i in tqdm(range(args.num_samples)):
        sig, mask, lbl = generate_segmentation_sample(i)
        current_signals.append(sig)
        current_masks.append(mask)
        current_labels.append(lbl)
        
        if len(current_signals) >= chunk_size:
            print(f"Saving chunk {chunk_idx}...")
            np.savez_compressed(
                os.path.join(args.output_dir, f'train_data_chunk_{chunk_idx}.npz'),
                signals=np.array(current_signals, dtype=np.float32)[:, np.newaxis, :],
                masks=np.array(current_masks, dtype=np.uint8),
                labels=np.array(current_labels, dtype=np.int64)
            )
            current_signals = []
            current_masks = []
            current_labels = []
            chunk_idx += 1
            
    # Save remaining
    if len(current_signals) > 0:
        print(f"Saving final chunk {chunk_idx}...")
        np.savez_compressed(
            os.path.join(args.output_dir, f'train_data_chunk_{chunk_idx}.npz'),
            signals=np.array(current_signals, dtype=np.float32)[:, np.newaxis, :],
            masks=np.array(current_masks, dtype=np.uint8),
            labels=np.array(current_labels, dtype=np.int64)
        )
            
    print("Done!")


if __name__ == '__main__':
    main()
