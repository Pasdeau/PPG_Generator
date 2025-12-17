"""
PPG Generator Module

Main PPG signal generation from RR intervals

Copyright (C) 2019 Andrius Solosenko (original MATLAB version)
Python conversion (C) 2024

PPG generator is freeware for private, non-commercial use.

Related publications:
- Solosenko, A., Petrenas, A., Marozas, V., & Sornmo, L. (2017). 
  Modeling of the photoplethysmogram during atrial fibrillation. 
  Computers in Biology and Medicine, 81, 130–138.
- Paliakaite, B., Petrenas, A., Solosenko, A., & Marozas, V. (2020). 
  Photoplethysmogram modeling of extreme bradycardia and ventricular tachycardia.
"""

import numpy as np
from scipy import interpolate, signal
from ppg_pulse import gen_PPGpulse, get_pulse_type_parameters


def gen_PPG(RR, pulse_type=1, Fd=1000):
    """
    Generate PPG signal from RR intervals
    
    This function generates a realistic PPG (photoplethysmogram) signal based on
    provided RR intervals. The PPG morphology is constructed using pulse templates
    that vary according to heart rate variability.
    
    Parameters:
    -----------
    RR : array-like
        RR intervals in milliseconds
    pulse_type : int
        Pulse morphology type (1-5)
        1 - Type 1 (normal), 2 - Type 2, 3 - Type 3, 4 - Type 3 bis, 5 - Type 4
    Fd : int
        Sampling frequency in Hz (default: 1000)
        
    Returns:
    --------
    PPGmodel : array
        Generated PPG signal
    PPGpeakIdx : array
        Sample indices where pulse peaks occur
    PPGpeakVal : array
        PPG amplitude values at peaks
        
    Examples:
    ---------
    >>> RR = np.array([800, 850, 820, 810, 840])  # RR intervals in ms
    >>> ppg, peak_idx, peak_val = gen_PPG(RR, pulse_type=1, Fd=1000)
    """
    RR = np.asarray(RR, dtype=float)
    num_beats = len(RR)
    
    # Convert RR intervals from milliseconds to samples
    RR_samples = np.round(RR * Fd / 1000).astype(int)
    
    # Calculate total signal length
    total_length = int(np.sum(RR_samples))
    
    # Initialize PPG signal
    PPGmodel = np.zeros(total_length)
    
    # Get pulse type parameters
    pulse_params = get_pulse_type_parameters(pulse_type)
    
    # Arrays to store peak information
    PPGpeakIdx = np.zeros(num_beats, dtype=int)
    PPGpeakVal = np.zeros(num_beats)
    
    # Current position in the signal
    current_pos = 0
    
    # CRITICAL: Track the baseline for beat-to-beat continuity
    # First beat can start at any baseline (we'll use the natural value from the model)
    # Subsequent beats will inherit the endpoint of the previous beat
    previous_endpoint = None
    
    # Generate PPG for each heartbeat
    for i in range(num_beats):
        # Duration of current beat
        beat_duration = RR_samples[i]
        
        # Create normalized time vector for this beat (0 to 1)
        t_norm = np.linspace(0, 1, beat_duration)
        
        # Generate base pulse shape with target baseline from previous beat
        if previous_endpoint is None:
            # First beat: let it determine its own baseline
            pulse = gen_PPGpulse(t_norm, pulse_params, target_baseline=None)
        else:
            # Subsequent beats: inherit baseline from previous beat
            pulse = gen_PPGpulse(t_norm, pulse_params, target_baseline=previous_endpoint)
        
        # Minimal heart rate dependent amplitude variation (more subtle)
        # This matches MATLAB's approach better
        hr = 60000 / RR[i]  # Heart rate in bpm
        hr_factor = 1.0 + 0.05 * (70 - hr) / 70  # Much smaller modulation
        hr_factor = np.clip(hr_factor, 0.9, 1.1)
        
        # Very minimal random variation to avoid identical beats
        amp_variation = 1.0 + 0.02 * np.random.randn()
        
        # Apply amplitude modulation
        pulse = pulse * hr_factor * amp_variation
        
        # CRITICAL: Re-enforce continuity after amplitude modulation
        # Ensure the pulse still starts at the target baseline
        if previous_endpoint is not None:
            # Adjust pulse to maintain the inherited baseline
            actual_start = pulse[0]
            offset = previous_endpoint - actual_start
            pulse = pulse + offset
            
            # Make sure end equals start again (amplitude modulation preserves this)
            baseline_diff = pulse[-1] - pulse[0]
            if abs(baseline_diff) > 1e-9:
                x_norm_fix = np.linspace(0, 1, len(pulse))
                correction = -baseline_diff * (3 * x_norm_fix**2 - 2 * x_norm_fix**3)
                pulse = pulse + correction
        
        # Store the endpoint for the next beat
        previous_endpoint = pulse[-1]
        
        # Find peak in current pulse
        peak_idx_local = np.argmax(pulse)
        peak_value = pulse[peak_idx_local]
        
        # Store peak information (global index)
        PPGpeakIdx[i] = current_pos + peak_idx_local
        PPGpeakVal[i] = peak_value
        
        # Insert pulse into PPG signal
        end_pos = current_pos + beat_duration
        PPGmodel[current_pos:end_pos] = pulse
        
        # Update position
        current_pos = end_pos
    
    # Add respiratory baseline wander (physiological low-frequency modulation)
    # Respiration typically occurs at 0.2-0.3 Hz (12-18 breaths per minute)
    # This creates the natural AC component that modulates the PPG signal
    signal_length = len(PPGmodel)
    duration_sec = signal_length / Fd
    t_signal = np.arange(signal_length) / Fd
    
    # PHYSIOLOGICAL COUPLING: Respiratory frequency based on heart rate
    # Typical ratio: Heart Rate / Respiratory Rate ≈ 4-5:1
    mean_hr = 60000 / np.mean(RR)  # Average heart rate in bpm
    base_resp_rate = mean_hr / 4.5  # Respiratory rate in breaths/min
    resp_freq = base_resp_rate / 60.0  # Convert to Hz
    
    # Add natural variation to respiratory frequency (±10%)
    resp_freq = resp_freq * (1 + 0.1 * np.random.randn())
    resp_freq = np.clip(resp_freq, 0.15, 0.35)  # Physiological limits: 9-21 breaths/min
    
    # Respiratory modulation: sinusoidal baseline wander
    # Amplitude: ~5-10% of signal range
    signal_range = np.max(PPGmodel) - np.min(PPGmodel)
    resp_amplitude = 0.075 * signal_range * (1 + 0.3 * np.random.randn())
    
    # Create respiratory wander with smooth phase start
    resp_phase = 2 * np.pi * np.random.rand()  # Random starting phase
    respiratory_wander = resp_amplitude * np.sin(2 * np.pi * resp_freq * t_signal + resp_phase)
    
    # Add secondary harmonic for more realistic breathing (not perfectly sinusoidal)
    resp_harmonic = 0.15 * resp_amplitude * np.sin(4 * np.pi * resp_freq * t_signal + resp_phase * 1.3)
    respiratory_wander = respiratory_wander + resp_harmonic
    
    # Add respiratory wander to PPG signal
    PPGmodel = PPGmodel + respiratory_wander
    
    # MINIMAL physiological noise for clean PPG (like MATLAB reference)
    # Note: Most "realistic" noise comes from respiratory modulation above
    # Additional high-frequency noise should be very subtle to avoid rough appearance
    
    # Very subtle high-frequency noise (0.1% instead of 1%)
    # This represents minimal sensor imperfection, not rough texture
    hf_noise_level = 0.001 * signal_range  # Reduced from 0.01 to 0.001
    hf_noise = hf_noise_level * np.random.randn(signal_length)
    
    # Add only minimal noise
    PPGmodel = PPGmodel + hf_noise
    
    # Update peak values to reflect all modulations
    for i in range(num_beats):
        PPGpeakVal[i] = PPGmodel[PPGpeakIdx[i]]
    
    # Log respiratory parameters (for debugging/validation)
    # print(f"Heart rate: {mean_hr:.1f} bpm, Respiratory rate: {base_resp_rate:.1f} breaths/min ({resp_freq:.3f} Hz)")
    
    return PPGmodel, PPGpeakIdx, PPGpeakVal


def detect_ppg_peaks(ppg_signal, Fd=1000, min_distance_ms=300):
    """
    Detect peaks in PPG signal
    
    Parameters:
    -----------
    ppg_signal : array
        PPG signal
    Fd : int
        Sampling frequency in Hz
    min_distance_ms : int
        Minimum distance between peaks in milliseconds
        
    Returns:
    --------
    peak_idx : array
        Indices of detected peaks
    peak_val : array
        Values at detected peaks
    """
    min_distance_samples = int(min_distance_ms * Fd / 1000)
    
    # Find peaks
    peaks, properties = signal.find_peaks(ppg_signal, 
                                         distance=min_distance_samples,
                                         prominence=0.1*np.std(ppg_signal))
    
    return peaks, ppg_signal[peaks]


if __name__ == "__main__":
    # Test the PPG generator
    import matplotlib.pyplot as plt
    
    # Generate test RR intervals
    num_beats = 20
    mean_rr = 800  # ms
    std_rr = 50    # ms
    
    # Sinus rhythm with some variability
    t = np.arange(num_beats)
    RR = mean_rr + std_rr * np.sin(2 * np.pi * 0.1 * t) + 20 * np.random.randn(num_beats)
    RR = np.clip(RR, 500, 1200)
    
    print(f"Generating PPG from {num_beats} RR intervals...")
    print(f"RR: mean={np.mean(RR):.1f}ms, std={np.std(RR):.1f}ms")
    
    # Generate PPG
    ppg, peak_idx, peak_val = gen_PPG(RR, pulse_type=1, Fd=1000)
    
    print(f"Generated PPG signal: {len(ppg)} samples ({len(ppg)/1000:.2f} seconds)")
    print(f"Detected {len(peak_idx)} peaks")
    
    # Plot
    t_ppg = np.arange(len(ppg)) / 1000  # Time in seconds
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot RR intervals
    axes[0].plot(np.arange(len(RR)), RR, 'o-', markersize=6, linewidth=2)
    axes[0].set_xlabel('Beat Number')
    axes[0].set_ylabel('RR Interval (ms)')
    axes[0].set_title('Input RR Intervals')
    axes[0].grid(True, alpha=0.3)
    
    # Plot PPG signal
    axes[1].plot(t_ppg, ppg, 'b-', linewidth=1, label='PPG Signal')
    axes[1].plot(peak_idx/1000, peak_val, 'ro', markersize=6, label='Detected Peaks')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Generated PPG Signal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_ppg_generator.png', dpi=150)
    print("Saved test plot: test_ppg_generator.png")
    plt.show()
