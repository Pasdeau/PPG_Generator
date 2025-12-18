#!/usr/bin/env python3
"""
Evaluate Trained Model Locally

This script loads the downloaded model and evaluates it on:
1. Waveform classification (5 types)
2. Noise robustness (by adding synthetic noise)

Usage:
    python evaluate_local.py --model_path local_best_model.pth
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ml_training.models import create_model
from ppg_generator import gen_PPG
from data_loader import generate_synthetic_rr

def generate_test_batch(batch_size=50, noise_level=0.0):
    """Generate a batch of synthetic data for testing"""
    signals = []
    labels = []
    
    for i in range(batch_size):
        # Evenly distribute pulse types
        pulse_type = (i % 5) + 1
        
        # Generate RR intervals
        RR = generate_synthetic_rr(num_beats=60, rhythm_type='SR', mean_rr=800, std_rr=50)
        
        # Generate PPG
        ppg, _, _ = gen_PPG(RR, pulse_type=pulse_type, Fd=1000)
        
        # Normalize
        ppg = (ppg - np.mean(ppg)) / np.std(ppg)
        
        # Add extra noise if requested
        if noise_level > 0:
            noise = np.random.randn(len(ppg)) * noise_level
            ppg = ppg + noise
        
        # Pad/Crop to 8000 samples
        target_len = 8000
        if len(ppg) < target_len:
            pad = target_len - len(ppg)
            ppg = np.pad(ppg, (0, pad), 'constant')
        else:
            ppg = ppg[:target_len]
            
        signals.append(ppg)
        labels.append(pulse_type - 1)  # 0-indexed labels
        
    return torch.FloatTensor(np.array(signals)).unsqueeze(1), torch.LongTensor(labels)

def evaluate(model, device):
    """Evaluate model on synthetic data"""
    model.eval()
    
    print("\nEvaluating Model Capabilities...")
    print("-" * 50)
    
    # Test 1: Clean Signals (Waveform Classification)
    print("\nTest 1: Waveform Classification (Clean Signals)")
    signals, labels = generate_test_batch(batch_size=100, noise_level=0.0)
    signals, labels = signals.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(signals)
        _, predicted = torch.max(outputs.data, 1)
        
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        
        print(f"Accuracy on Clean Signals: {100 * correct / total:.2f}%")
        
        # Per-class accuracy
        for i in range(5):
            mask = labels == i
            if mask.sum() > 0:
                class_acc = (predicted[mask] == labels[mask]).sum().item() / mask.sum().item()
                print(f"  Type {i+1}: {100 * class_acc:.1f}%")

    # Test 2: Noise Robustness
    print("\nTest 2: Noise Robustness (Adding Gaussian Noise)")
    noise_levels = [0.1, 0.5, 1.0, 2.0]
    
    for nl in noise_levels:
        signals, labels = generate_test_batch(batch_size=100, noise_level=nl)
        signals, labels = signals.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)
            acc = 100 * (predicted == labels).sum().item() / labels.size(0)
            print(f"  Noise Level {nl} (SNR ~{-20*np.log10(nl):.1f}dB): {acc:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    # Note: We need to know the architecture used. Assuming ResNet1D as per logs.
    print(f"Loading model from {args.model_path}")
    
    # We need to recreate the model structure first
    # Based on training logs, it was a ResNet1D for waveform task (5 classes)
    model = create_model('resnet', input_length=8000, num_classes=5)
    
    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle state dict key mismatch if wrapped in DataParallel or similar
    state_dict = checkpoint['model_state_dict']
    
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Direct load failed, trying to adjust keys: {e}")
        # Try removing 'module.' prefix if present
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        
    model = model.to(device)
    print("Model loaded successfully!")
    
    evaluate(model, device)

if __name__ == '__main__':
    main()
