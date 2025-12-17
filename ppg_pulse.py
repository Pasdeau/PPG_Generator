"""
PPG Pulse Generation Module

This module generates PPG pulse shapes using a combination of lognormal 
and Gaussian functions.

Copyright (C) 2019 Andrius Solosenko (original MATLAB version)
Python conversion (C) 2024

Based on:
gen_PPGpulse.m from the original MATLAB PPG generator
"""

import numpy as np
from scipy.stats import lognorm


def gaussmf(x, params):
    """
    Gaussian membership function (equivalent to MATLAB's gaussmf)
    
    Parameters:
    -----------
    x : array-like
        Input values
    params : list [sigma, center]
        sigma: standard deviation
        center: center of the Gaussian
        
    Returns:
    --------
    array-like
        Gaussian function values
    """
    sigma, center = params
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def get_pulse_type_parameters(pulse_type):
    """
    Get parameters for different PPG pulse types
    
    Parameters:
    -----------
    pulse_type : int
        1 - Type 1, 2 - Type 2, 3 - Type 3, 4 - Type 3 bis, 5 - Type 4
        
    Returns:
    --------
    dict
        Dictionary containing pulse parameters (k0, tau0, mu0, sigma0, k1, mu1, sigma1, k2, mu2, sigma2)
    """
    # Parameters based on the literature and empirical observations
    # These represent different PPG morphologies seen in practice
    
    if pulse_type == 1:  # Type 1 - MATLAB fitted
        params = {
            'k0': 1.083445,
            'tau0': 0.080122,
            'mu0': -0.000000,
            'sigma0': 0.407575,
            'k1': 0.779456,
            'sigma1': 0.242535,  # Restore original fitted value
            'mu1': 0.000000,
            'k2': -1.000000,
            'sigma2': 0.115955,
            'mu2': 0.746101,
        }
    elif pulse_type == 2:  # Type 2 - MATLAB fitted
        params = {
            'k0': 0.987453,
            'tau0': 0.049877,
            'mu0': -0.000000,
            'sigma0': 0.353608,
            'k1': 0.849010,
            'sigma1': 0.273403,  # Restore original fitted value
            'mu1': 0.000000,
            'k2': -1.000000,
            'sigma2': 0.113131,
            'mu2': 0.758938,
        }
    elif pulse_type == 3:  # Type 3 - MATLAB fitted
        params = {
            'k0': 0.543502,
            'tau0': 0.500000,
            'mu0': -0.516995,
            'sigma0': 0.390013,
            'k1': 0.873918,
            'sigma1': 0.276442,  # Restore original fitted value
            'mu1': 0.000000,
            'k2': -0.215151,
            'sigma2': 0.054597,
            'mu2': 0.773650,
        }
    elif pulse_type == 4:  # Type 4 - MATLAB fitted
        params = {
            'k0': 0.868647,
            'tau0': 0.498306,
            'mu0': -0.765884,
            'sigma0': 0.432124,
            'k1': 0.896343,
            'sigma1': 0.268131,  # Restore original fitted value
            'mu1': 0.000000,
            'k2': -1.000000,
            'sigma2': 0.125214,
            'mu2': 0.866838,
        }
    elif pulse_type == 5:  # Type 5 - MATLAB fitted
        params = {
            'k0': 0.854360,
            'tau0': 0.347632,
            'mu0': -0.245623,
            'sigma0': 0.500000,
            'k1': 0.915304,
            'sigma1': 0.251524,  # Restore original fitted value
            'mu1': 0.000000,
            'k2': -0.512745,
            'sigma2': 0.086594,
            'mu2': 0.692531,
        }
    else:
        raise ValueError(f"Invalid pulse_type: {pulse_type}. Must be 1, 2, 3, 4, or 5")
    
    return params


def gen_PPGpulse(t, pulse_type, target_baseline=None):
    """
    Generate PPG pulse waveform using lognormal and Gaussian functions.
    
    Parameters
    ----------
    t : array_like
        Normalized time vector (typically 0 to 1)
    pulse_type : int or dict
        Pulse type number (1-5) or parameter dictionary
    target_baseline : float, optional
        Target baseline value for pulse endpoints. If None, pulse determines
        its own baseline. If specified, pulse endpoints will be forced to this value.
        This enables beat-to-beat baseline inheritance for natural continuity.
    
    Returns
    -------
    P : ndarray
        PPG pulse waveform
        
    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> pulse = gen_PPGpulse(t, 1)
    >>> pulse_inherited = gen_PPGpulse(t, 1, target_baseline=0.5)
    """
    # Convert to numpy array if not already
    t = np.asarray(t)
    
    # Get parameters
    if isinstance(pulse_type, dict):
        params = pulse_type
    else:
        params = get_pulse_type_parameters(pulse_type)
    
    # Extract parameters
    k0 = params['k0']
    tau0 = params['tau0']
    mu0 = params['mu0']
    sigma0 = params['sigma0']
    k1 = params['k1']
    sigma1 = params['sigma1']
    mu1 = params['mu1']
    k2 = params['k2']
    sigma2 = params['sigma2']
    mu2 = params['mu2']
    
    # Calculate lognormal component (g0)
    # Note: scipy's lognorm uses a different parameterization than MATLAB
    # MATLAB: lognpdf(x-tau, mu, sigma)
    # scipy: lognorm(s=sigma, scale=exp(mu)).pdf(x-tau)
    t_shifted = t - tau0
    # Handle negative values (lognormal only defined for positive values)
    g0 = np.zeros_like(t)
    positive_mask = t_shifted > 0
    if np.any(positive_mask):
        g0[positive_mask] = k0 * lognorm(s=sigma0, scale=np.exp(mu0)).pdf(t_shifted[positive_mask])
    
    # Calculate Gaussian components (g1 and g2)
    g1 = k1 * gaussmf(t, [sigma1, mu1])
    g2 = k2 * gaussmf(t, [sigma2, mu2])
    
    # Sum of components
    P = g0 + g1 + g2
    
    # CRITICAL FIX: Ensure beat-to-beat continuity
    # Strategy: 
    #   - If target_baseline is None: force endpoints to match each other (self-continuous)
    #   - If target_baseline is provided: force endpoints to that value (inherit from previous beat)
    
    if target_baseline is None:
        # First beat or standalone pulse: make it self-continuous
        # Force both endpoints to their average
        current_avg = (P[0] + P[-1]) / 2
        P = P - current_avg  # Shift to zero average
        
        # Make endpoints exactly equal using cubic correction
        baseline_diff = P[-1] - P[0]
        if abs(baseline_diff) > 1e-12:
            x_norm = np.linspace(0, 1, len(t))
            correction = -baseline_diff * (3 * x_norm**2 - 2 * x_norm**3)
            P = P + correction
    else:
        # Subsequent beat: inherit baseline from previous beat
        # Shift pulse so that both endpoints equal target_baseline
        current_avg = (P[0] + P[-1]) / 2
        P = P - current_avg + target_baseline
        
        # Fine-tune to ensure exact match using cubic correction
        baseline_diff = P[-1] - P[0]
        if abs(baseline_diff) > 1e-12:
            x_norm = np.linspace(0, 1, len(t))
            correction = -baseline_diff * (3 * x_norm**2 - 2 * x_norm**3)
            P = P + correction
    
    return P


if __name__ == "__main__":
    # Test the function
    import matplotlib.pyplot as plt
    
    t = np.linspace(0, 1, 1000)
    
    plt.figure(figsize=(12, 8))
    for i in range(1, 6):
        plt.subplot(3, 2, i)
        pulse = gen_PPGpulse(t, i)
        plt.plot(t, pulse, 'b-', linewidth=2)
        plt.title(f'PPG Pulse Type {i}')
        plt.xlabel('Normalized Time')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ppg_pulse_types.png', dpi=150)
    print("Generated test plot: ppg_pulse_types.png")
    plt.show()
