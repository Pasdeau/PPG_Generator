#!/usr/bin/env python3
"""
Python PPG Signal Generator - Main Program

Simplified Python version of MATLAB PPG generation system.
Generates PPG signals from RR intervals with optional artifacts.

Usage:
    python main_ppg.py

Copyright (C) 2024
Based on original MATLAB code by Andrius Solosenko and Birute Paliakaite
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
import os
from ppg_generator import gen_PPG
from ppg_artifacts import gen_PPG_artifacts
from data_loader import load_artifact_params, generate_synthetic_rr, save_ppg_data


def main():
    """Main PPG generation program"""
    
    print("=" * 60)
    print("PPG Signal Generator (Python Version)")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description='PPG Signal Generator')
    parser.add_argument('--num_beats', type=int, default=10, help='Number of beats')
    parser.add_argument('--pulse_type', type=int, default=1, help='Pulse type (1-5)')
    parser.add_argument('--no_artifacts', action='store_true', help='Disable artifacts')
    parser.add_argument('--dur_mu0', type=float, default=15.0, help='Artifact-free duration (s)')
    parser.add_argument('--artifact_type', type=int, default=2, help='Artifact type (1-4)')
    parser.add_argument('--dur_mu', type=float, default=2.0, help='Artifact duration (s)')
    args = parser.parse_args()

    # Output settings
    output_dir = 'output'
    output_prefix = 'python_ppg_test'
    save_npz = True
    save_csv = False
    
    os.makedirs(output_dir, exist_ok=True)
    
    # PPG Generation Parameters
    num_beats = args.num_beats
    pulse_type = args.pulse_type
    Fd = 1000
    
    # RR Interval Parameters
    use_synthetic_rr = True
    rhythm_type = 'SR'
    mean_rr = 800
    std_rr = 50
    
    # Artifact Parameters
    add_artifacts = not args.no_artifacts
    # Create one-hot vector for artifact type
    typ_artifact = np.zeros(4)
    if 1 <= args.artifact_type <= 4:
        typ_artifact[args.artifact_type - 1] = 1
    else:
        typ_artifact[1] = 1 # Default to type 2
        
    dur_mu0 = args.dur_mu0
    dur_mu = args.dur_mu
    
    print(f"\nConfiguration:")
    print(f"  Number of beats: {num_beats}")
    print(f"  Pulse type: {pulse_type}")
    print(f"  Sampling frequency: {Fd} Hz")
    print(f"  Rhythm type: {rhythm_type}")
    print(f"  Add artifacts: {add_artifacts} (Interval: {dur_mu0}s)")
    
    # ----- Generate or Load RR Intervals -----
    
    print(f"\n{'=' * 60}")
    print("Step 1: Generate RR Intervals")
    print(f"{'=' * 60}")
    
    if use_synthetic_rr:
        RR = generate_synthetic_rr(num_beats, rhythm_type=rhythm_type, 
                                   mean_rr=mean_rr, std_rr=std_rr)
        print(f"✓ Generated {len(RR)} synthetic {rhythm_type} RR intervals")
    else:
        # You can uncomment and modify this to load real RR data
        # from data_loader import load_rr_intervals
        # rr_file = 'DATA_RR_SR_real.mat'
        # RR = load_rr_intervals(rr_file, rhythm_type='SR', max_beats=num_beats)
        print("Loading from file not configured. Using synthetic data instead.")
        RR = generate_synthetic_rr(num_beats, rhythm_type=rhythm_type, 
                                   mean_rr=mean_rr, std_rr=std_rr)
    
    print(f"  RR statistics: mean={np.mean(RR):.1f}ms, std={np.std(RR):.1f}ms")
    print(f"  Heart rate: {60000/np.mean(RR):.1f} bpm")
    
    # ----- Generate PPG Signal -----
    
    print(f"\n{'=' * 60}")
    print("Step 2: Generate PPG Signal")
    print(f"{'=' * 60}")
    
    PPGmodel, PPGpeakIdx, PPGpeakVal = gen_PPG(RR, pulse_type=pulse_type, Fd=Fd)
    
    print(f"✓ Generated PPG signal")
    print(f"  Duration: {len(PPGmodel)/Fd:.2f} seconds ({len(PPGmodel)} samples)")
    print(f"  Detected {len(PPGpeakIdx)} peaks")
    
    # ----- Generate Artifacts -----
    
    print(f"\n{'=' * 60}")
    print("Step 3: Generate Artifacts")
    print(f"{'=' * 60}")
    
    if add_artifacts and np.sum(typ_artifact) > 0:
        # Load artifact parameters
        artifact_params = load_artifact_params('artifact_param.mat')
        
        # Calculate probabilities
        prob = typ_artifact * (1 / np.sum(typ_artifact))
        
        # Duration parameters
        dur_mu_vec = np.array([dur_mu0] + [dur_mu] * 4)
        
        # Generate PSD slopes for each artifact type
        slope = np.zeros(4)
        for n in range(4):
            slope[n] = artifact_params['slope_sd'][n] * np.random.randn() + artifact_params['slope_m'][n]
        
        # Generate artifacts
        artifact = gen_PPG_artifacts(
            len(PPGmodel), prob, dur_mu_vec,
            artifact_params['RMS_shape'], artifact_params['RMS_scale'],
            slope, Fd
        )
        
        print(f"✓ Generated artifacts")
        print(f"  Artifact RMS: {np.std(artifact):.4f}")
    else:
        artifact = np.zeros_like(PPGmodel)
        print("✓ No artifacts added")
    
    # ----- Combine PPG and Artifacts -----
    
    print(f"\n{'=' * 60}")
    print("Step 4: Combine PPG and Artifacts")
    print(f"{'=' * 60}")
    
    # Normalize pulsatile component
    PPGmodel_norm = (PPGmodel - np.mean(PPGmodel)) / np.std(PPGmodel)
    
    # Combine
    PPG = PPGmodel_norm + artifact
    
    print(f"✓ Combined PPG and artifacts")
    print(f"  Final signal RMS: {np.std(PPG):.4f}")
    
    # ----- Visualization -----
    
    print(f"\n{'=' * 60}")
    print("Step 5: Create Visualizations")
    print(f"{'=' * 60}")
    
    # Create time vectors
    t_ppg = np.arange(len(PPG)) / Fd
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Complete PPG signal
    axes[0].plot(t_ppg, PPG, 'k-', linewidth=0.8)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('PPG Signal (Pulsatile + Artifacts)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Pulsatile component only
    axes[1].plot(t_ppg, PPGmodel_norm, 'b-', linewidth=0.8)
    # Use normalized values for peak marking
    PPGpeakVal_norm = PPGmodel_norm[PPGpeakIdx]
    axes[1].plot(PPGpeakIdx/Fd, PPGpeakVal_norm, 'ro', markersize=5, label='Peaks')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Pulsatile Component')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Artifacts only
    axes[2].plot(t_ppg, artifact, 'r-', linewidth=0.8)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Artifacts')
    axes[2].grid(True, alpha=0.3)
    
    # Link x-axes
    for ax in axes:
        ax.set_xlim([0, t_ppg[-1]])
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, f'{output_prefix}_PPG.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved figure: {fig_path}")
    
    # Also create RR interval plot
    fig2, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(RR)), RR, 'o-', markersize=6, linewidth=2, color='#2E86AB')
    ax.set_xlabel('Beat Number', fontsize=12)
    ax.set_ylabel('RR Interval (ms)', fontsize=12)
    ax.set_title(f'RR Intervals ({rhythm_type})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig2_path = os.path.join(output_dir, f'{output_prefix}_RR.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved figure: {fig2_path}")
    
    # FFT Analysis
    fig3, axes3 = plt.subplots(2, 1, figsize=(14, 8))
    
    # Compute FFT
    fft_result = np.fft.rfft(PPG)
    freq = np.fft.rfftfreq(len(PPG), 1/Fd)
    magnitude = np.abs(fft_result)
    power_spectrum = magnitude ** 2
    
    # Find top 4 peaks using scipy peak detection
    from scipy.signal import find_peaks
    
    # Find peaks in positive frequency range (exclude DC and very low frequencies)
    freq_mask = (freq > 0.1) & (freq < 10)  # Focus on 0.1-10 Hz
    freq_subset = freq[freq_mask]
    mag_subset = magnitude[freq_mask]
    
    # Find peaks with minimum distance between peaks
    peak_indices, peak_properties = find_peaks(mag_subset, distance=int(len(mag_subset)/20))
    
    # Sort by magnitude and get top 4
    if len(peak_indices) > 0:
        peak_magnitudes = mag_subset[peak_indices]
        sorted_idx = np.argsort(peak_magnitudes)[::-1]  # Descending order
        top_4_idx = sorted_idx[:min(4, len(sorted_idx))]
        top_4_peaks = [(freq_subset[peak_indices[i]], mag_subset[peak_indices[i]]) for i in top_4_idx]
    else:
        top_4_peaks = []
    
    # Plot 1: FFT Magnitude with top 4 peaks
    axes3[0].plot(freq, magnitude, 'b-', linewidth=1)
    
    # Annotate top 4 peaks
    colors = ['red', 'orange', 'green', 'purple']
    for i, (peak_freq, peak_mag) in enumerate(top_4_peaks):
        peak_hr = peak_freq * 60  # Convert to bpm
        axes3[0].plot(peak_freq, peak_mag, 'o', color=colors[i], markersize=10, 
                     label=f'Peak {i+1}: {peak_freq:.2f} Hz ({peak_hr:.1f} bpm)')
        axes3[0].annotate(f'{peak_freq:.2f} Hz\n({peak_mag:.1f})', 
                         xy=(peak_freq, peak_mag),
                         xytext=(peak_freq + 0.3, peak_mag * (1.1 - 0.15*i)),
                         fontsize=9, color=colors[i],
                         arrowprops=dict(arrowstyle='->', color=colors[i], lw=0.8))
    
    axes3[0].set_xlabel('Frequency (Hz)', fontsize=12)
    axes3[0].set_ylabel('Magnitude', fontsize=12)
    axes3[0].set_title('FFT Magnitude Spectrum (Top 4 Peaks)', fontsize=14, fontweight='bold')
    axes3[0].grid(True, alpha=0.3)
    axes3[0].legend(loc='upper right', fontsize=9)
    axes3[0].set_xlim([0, min(10, freq[-1])])  # Focus on 0-10 Hz
    
    # Plot 2: Power Spectrum (log scale) with peaks
    axes3[1].semilogy(freq, power_spectrum, 'g-', linewidth=1)
    for i, (peak_freq, peak_mag) in enumerate(top_4_peaks):
        # Find corresponding power value
        peak_power = power_spectrum[np.argmin(np.abs(freq - peak_freq))]
        axes3[1].plot(peak_freq, peak_power, 'o', color=colors[i], markersize=8)
    axes3[1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes3[1].set_ylabel('Power (log scale)', fontsize=12)
    axes3[1].set_title('Power Spectrum', fontsize=14, fontweight='bold')
    axes3[1].grid(True, alpha=0.3)
    axes3[1].set_xlim([0, min(10, freq[-1])])
    
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, f'{output_prefix}_FFT.png')
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved FFT figure: {fig3_path}")
    
    # Note: plt.show() removed for non-interactive execution


    
    # ----- Save Data -----
    
    print(f"\n{'=' * 60}")
    print("Step 6: Save Output Data")
    print(f"{'=' * 60}")
    
    if save_npz:
        npz_path = os.path.join(output_dir, output_prefix)
        save_ppg_data(npz_path, PPG, PPGmodel_norm, artifact, 
                     rr=RR, ppg_peak_idx=PPGpeakIdx, ppg_peak_val=PPGpeakVal,
                     sampling_freq=Fd)
    
    if save_csv:
        import csv
        csv_path = os.path.join(output_dir, f'{output_prefix}_PPG.csv')
        np.savetxt(csv_path, PPG, delimiter=',', fmt='%.6f')
        print(f"✓ Saved CSV: {csv_path}")
        
        rr_csv_path = os.path.join(output_dir, f'{output_prefix}_RR.csv')
        np.savetxt(rr_csv_path, RR, delimiter=',', fmt='%.2f')
        print(f"✓ Saved RR CSV: {rr_csv_path}")
    
    print(f"\n{'=' * 60}")
    print("✓ PPG Generation Complete!")
    print(f"{'=' * 60}")
    print(f"\nOutput files saved to: {output_dir}/")
    print(f"  - {output_prefix}_PPG.png")
    print(f"  - {output_prefix}_RR.png")
    print(f"  - {output_prefix}_FFT.png")
    if save_npz:
        print(f"  - {output_prefix}.npz")
    if save_csv:
        print(f"  - {output_prefix}_PPG.csv")
        print(f"  - {output_prefix}_RR.csv")
    
    return PPG, PPGmodel_norm, artifact, RR


if __name__ == "__main__":
    PPG, PPGmodel, artifact, RR = main()
