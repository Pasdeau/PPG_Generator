"""
PPG Artifacts Generation Module

Generates realistic PPG artifacts including:
1. Device displacement
2. Forearm motion  
3. Hand motion
4. Poor contact

Copyright (C) 2020 Birute Paliakaite (original MATLAB version)
Python conversion (C) 2024

Available under the GNU General Public License version 3

Related publication: Paliakaite, B., Petrenas, A., Solosenko, A., & Marozas, V. (2021). 
Modeling of artifacts in the wrist photoplethysmogram: Application to the detection of 
life-threatening arrhythmias. Biomedical Signal Processing and Control, 66, 102421.
https://doi.org/10.1016/j.bspc.2021.102421
"""

import numpy as np
from scipy import signal


def gen_PPG_artifacts(duration_sampl, prob, dur_mu, RMS_shape, RMS_scale, slope, Fd):
    """
    Generate PPG artifacts
    
    Parameters:
    -----------
    duration_sampl : int
        Duration of signal to generate in samples
    prob : array-like, shape (4,)
        Probabilities of artifacts [device_disp, forearm_motion, hand_motion, poor_contact]
    dur_mu : array-like, shape (5,)
        Mean duration of artifact-free and artifact intervals in seconds
        [artifact_free, device_disp, forearm_motion, hand_motion, poor_contact]
    RMS_shape : array-like, shape (4,)
        Shape parameter for gamma distribution of normalized RMS artifact amplitudes
    RMS_scale : array-like, shape (4,)
        Scale parameter for gamma distribution of normalized RMS artifact amplitudes
    slope : array-like, shape (4,)
        Generated artifact PSD slope for each artifact type
    Fd : int
        Sampling frequency in Hz
        
    Returns:
    --------
    signal_out : array
        Generated artifact signal
        
    Notes:
    ------
    Artifact types in vectors are in the following order:
    0 - device displacement
    1 - forearm motion
    2 - hand motion
    3 - poor contact
    
    The function models transitions between artifact-free intervals and artifacts 
    using a Markov chain. Transition from artifact-free interval to all four 
    artifact types is possible; however, only transition to artifact-free 
    interval is allowed from an artifact.
    """
    # Convert inputs to numpy arrays
    prob = np.asarray(prob)
    dur_mu = np.asarray(dur_mu)
    RMS_shape = np.asarray(RMS_shape)
    RMS_scale = np.asarray(RMS_scale)
    slope = np.asarray(slope)
    
    # Initialize empty vectors
    signal_out = np.array([])
    states_vec = np.array([])
    
    # Convert mean interval duration to samples
    dur_mu = dur_mu * Fd
    
    # Design filters with predefined slopes for each artifact type
    fv = np.linspace(0, 1, 100)  # Frequency vector for filter design
    b = np.zeros((4, 251))  # Filter coefficients (251 taps)
    
    for n in range(4):
        # Create desired frequency response with specified slope
        a = np.zeros(100)
        a[0:2] = 0  # Zero at DC
        # Apply slope to frequencies from index 2 onwards
        freq_hz = fv[2:] * (Fd / 2)
        a[2:] = slope[n] * np.log10(freq_hz)
        # Convert from dB to linear scale
        a = np.sqrt(10 ** (a / 10))
        # Design FIR filter using least squares
        b[n, :] = signal.firls(251, fv, a)
    
    # Stochastic transition matrix
    # States: 0 - artifact-free, 1 - device disp, 2 - forearm, 3 - hand, 4 - poor contact
    # P[i, j] = probability of transitioning from state i to state j
    P = np.zeros((5, 5))
    P[0, :] = np.concatenate([[0], prob])  # From artifact-free to any artifact
    P[1:, 0] = 1  # From any artifact to artifact-free
    P[1:, 1:] = 0  # No direct transitions between artifact types
    
    # Start with artifact-free interval (state 0)
    X = 0
    
    # Generate signal for defined duration
    while len(states_vec) < duration_sampl:
        # Duration of current interval (exponential distribution)
        T = int(round(np.random.exponential(dur_mu[X])))
        
        # Minimum interval is 1 second
        if T < Fd:
            T = Fd
        
        # Adjust if less than 1 second left to achieve duration
        if duration_sampl - (len(states_vec) + T) < Fd:
            T = duration_sampl - len(states_vec)
        
        if X > 0:  # If state has an artifact
            # Generate artifact by filtering white noise
            tr = signal.lfilter(b[X-1, :], 1, np.random.randn(T + 200))
            tr = tr[200:]  # Remove transient
            
            # Standardize
            tr = (tr - np.mean(tr)) / np.std(tr)
            
            # Generate corresponding RMS level from gamma distribution
            R = np.random.gamma(RMS_shape[X-1], RMS_scale[X-1])
            
            # Tune amplitude
            tr = R * tr
            
            # Add to signal
            signal_out = np.concatenate([signal_out, tr])
        else:  # Artifact-free state
            # Add zeros
            signal_out = np.concatenate([signal_out, np.zeros(T)])
        
        # Record current state
        states_vec = np.concatenate([states_vec, (X) * np.ones(T)])
        
        # Generate next state
        u = np.random.rand()
        X = np.argmax(u < np.cumsum(P[X, :]))
    
    # Trim to exact duration
    signal_out = signal_out[:duration_sampl]
    states_vec = states_vec[:duration_sampl]
    
    return signal_out


if __name__ == "__main__":
    # Test the function
    import matplotlib.pyplot as plt
    
    # Test parameters
    Fd = 1000  # 1000 Hz sampling
    duration = 30  # 30 seconds
    duration_sampl = duration * Fd
    
    # Artifact probabilities (equal probability for demonstration)
    prob = np.array([0.25, 0.25, 0.25, 0.25])
    
    # Mean durations in seconds
    dur_mu0 = 3  # Artifact-free intervals
    dur_mu = 2   # Artifact intervals
    dur_mu_vec = np.array([dur_mu0, dur_mu, dur_mu, dur_mu, dur_mu])
    
    # Gamma distribution parameters (example values)
    RMS_shape = np.array([2.0, 2.5, 3.0, 2.2])
    RMS_scale = np.array([0.5, 0.6, 0.7, 0.55])
    
    # PSD slopes (example values)
    slope = np.array([-1.5, -2.0, -2.5, -1.8])
    
    # Generate artifacts
    artifact = gen_PPG_artifacts(duration_sampl, prob, dur_mu_vec, 
                                 RMS_shape, RMS_scale, slope, Fd)
    
    # Plot result
    t = np.arange(len(artifact)) / Fd
    plt.figure(figsize=(14, 5))
    plt.plot(t, artifact, 'k-', linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Generated PPG Artifacts')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_artifacts.png', dpi=150)
    print(f"Generated {duration}s of artifacts")
    print(f"Artifact RMS: {np.std(artifact):.4f}")
    plt.show()
