import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import signal

# Ensure local modules are found
sys.path.append(os.getcwd())

from ppg_artifacts import gen_PPG_artifacts
from data_loader import load_artifact_params

def compare_artifact_types():
    print("[-] comparing 4 artifact types...")
    
    # 1. Setup Parameters (Load from MAT file now!)
    Fd = 1000
    duration_sec = 10
    duration_sampl = duration_sec * Fd
    
    print("[-] Loading real MATLAB parameters...")
    params = load_artifact_params() # Uses default 'data/artifact_param.mat'
    
    types = ['1: Device Disp', '2: Forearm Motion', '3: Hand Motion', '4: Poor Contact']
    colors = ['#FF4136', '#2ECC40', '#0074D9', '#FF851B']
    
    results = []
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2)
    
    # Generate each type
    for i in range(4):
        # Force specific type
        prob = np.zeros(4)
        prob[i] = 1.0 
        
        # Durations: Make it continuous noise for the whole 10s
        dur_mu = np.array([0.1, 100, 100, 100, 100]) # Long artifact duration
        
        # Random slope instance
        slope = np.zeros(4)
        slope[i] = params['slope_m'][i] # Use mean slope to represent the class ideal
        
        artifact = gen_PPG_artifacts(
            duration_sampl, 
            prob, 
            dur_mu,
            params['RMS_shape'], 
            params['RMS_scale'], 
            slope, 
            Fd
        )
        
        # Metrics
        rms = np.std(artifact)
        results.append((types[i], rms))
        
        # Plot Time Domain
        ax = fig.add_subplot(gs[i // 2, i % 2])
        t = np.arange(len(artifact)) / Fd
        ax.plot(t, artifact, color=colors[i], linewidth=0.8)
        ax.set_title(f"{types[i]}\nRMS: {rms:.2f} | Slope: {params['slope_m'][i]}", fontsize=11)
        # ax.set_ylim(-10, 10) # Removed fixed scale to adapt to large amplitudes (Type 1)
        ax.grid(True, alpha=0.3)
        if i >= 2: ax.set_xlabel("Time (s)")
        
        # Store for PSD comparison
        if i == 0:
            all_artifacts = [artifact]
        else:
            all_artifacts.append(artifact)

    # Plot PSD Comparison
    ax_psd = fig.add_subplot(gs[2, :])
    for i in range(4):
        f, Pxx = signal.welch(all_artifacts[i], Fd, nperseg=1024)
        ax_psd.semilogy(f, Pxx, label=types[i], color=colors[i], linewidth=1.5)
        
    ax_psd.set_title("Power Spectral Density Comparison (Frequency Content)", fontweight='bold')
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power/Frequency (dB/Hz)")
    ax_psd.set_xlim(0.1, 20) # Focus on physiological range
    ax_psd.legend()
    ax_psd.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    os.makedirs("validation", exist_ok=True)
    out_path = "validation/artifact_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"[+] Comparison saved to {out_path}")
    
    print("\n" + "="*45)
    print("    ARTIFACT STATISTICAL SUMMARY")
    print("="*45)
    print(f"{'Type':<20} | {'RMS (Amp)':<10} | {'Slope (Freq)'}")
    print("-" * 45)
    for i, (name, rms) in enumerate(results):
        print(f"{name:<20} | {rms:<10.2f} | {params['slope_m'][i]}")
    print("="*45)

if __name__ == "__main__":
    compare_artifact_types()
