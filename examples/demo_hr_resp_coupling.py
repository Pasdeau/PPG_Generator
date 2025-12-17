#!/usr/bin/env python3
"""
Demonstrate the relationship between heart rate and respiratory rate
"""

import numpy as np
import matplotlib.pyplot as plt
from ppg_generator import gen_PPG

# Test different heart rates
test_cases = [
    {"mean_rr": 1000, "name": "Slow HR (60 bpm)"},
    {"mean_rr": 800, "name": "Normal HR (75 bpm)"},
    {"mean_rr": 667, "name": "Fast HR (90 bpm)"},
]

fig, axes = plt.subplots(len(test_cases), 2, figsize=(16, 12))

for idx, case in enumerate(test_cases):
    mean_rr = case["mean_rr"]
    name = case["name"]
    
    # Generate RR intervals
    num_beats = 20
    RR = np.ones(num_beats) * mean_rr + 50 * np.random.randn(num_beats)
    RR = np.clip(RR, mean_rr - 100, mean_rr + 100)
    
    # Generate PPG
    PPG, peaks_idx, peaks_val = gen_PPG(RR, pulse_type=1, Fd=1000)
    
    # Calculate actual rates
    actual_hr = 60000 / np.mean(RR)
    expected_resp = actual_hr / 4.5
    
    # Time vector
    duration = len(PPG) / 1000
    t = np.arange(len(PPG)) / 1000
    
    # Plot signal
    ax1 = axes[idx, 0]
    ax1.plot(t, PPG, 'b-', linewidth=1, alpha=0.8)
    ax1.plot(peaks_idx / 1000, peaks_val, 'ro', markersize=4)
    ax1.set_title(f'{name}\nHR: {actual_hr:.1f} bpm, Expected Resp: {expected_resp:.1f} breaths/min ({expected_resp/60:.3f} Hz)',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Plot FFT
    ax2 = axes[idx, 1]
    fft_vals = np.fft.fft(PPG)
    fft_freq = np.fft.fftfreq(len(PPG), 1/1000)
    
    # Only positive frequencies up to 5 Hz
    mask = (fft_freq >= 0) & (fft_freq <= 5)
    fft_freq_pos = fft_freq[mask]
    fft_magnitude = np.abs(fft_vals[mask])
    
    ax2.plot(fft_freq_pos, fft_magnitude, 'b-', linewidth=1)
    
    # Mark expected respiratory peak
    resp_freq_hz = expected_resp / 60
    ax2.axvline(x=resp_freq_hz, color='green', linestyle='--', linewidth=2, 
                label=f'Expected Resp: {resp_freq_hz:.3f} Hz')
    
    # Mark heart rate fundamental
    hr_freq_hz = actual_hr / 60
    ax2.axvline(x=hr_freq_hz, color='red', linestyle='--', linewidth=2,
                label=f'HR Fundamental: {hr_freq_hz:.3f} Hz')
    
    ax2.set_title(f'FFT Spectrum - HR:Resp Ratio ≈ {actual_hr/expected_resp:.1f}:1', 
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlim(0, 5)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/hr_resp_coupling_demo.png', dpi=150)
print("\n✓ Saved: output/hr_resp_coupling_demo.png")
print("\nDemonstration of HR-Respiratory coupling (4.5:1 ratio)")
print("=" * 70)
