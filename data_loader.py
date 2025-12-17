"""
Data Loading Module for PPG Generation

Handles loading of MATLAB .mat files containing parameters and data

Copyright (C) 2024
"""

import numpy as np
import scipy.io as sio
import os


def load_artifact_params(filepath='artifact_param.mat'):
    """
    Load artifact parameters from .mat file
    
    Parameters:
    -----------
    filepath : str
        Path to artifact_param.mat file
        
    Returns:
    --------
    dict
        Dictionary containing:
        - RMS_shape : array, shape (4,) - Shape parameter for gamma distribution
        - RMS_scale : array, shape (4,) - Scale parameter for gamma distribution
        - slope_m : array, shape (4,) - Mean of estimated PSD slopes
        - slope_sd : array, shape (4,) - Standard deviation of estimated PSD slopes
    """
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using default parameters.")
        # Return default parameters
        return {
            'RMS_shape': np.array([2.0, 2.5, 3.0, 2.2]),
            'RMS_scale': np.array([0.5, 0.6, 0.7, 0.55]),
            'slope_m': np.array([-1.5, -2.0, -2.5, -1.8]),
            'slope_sd': np.array([0.3, 0.35, 0.4, 0.32])
        }
    
    try:
        mat_data = sio.loadmat(filepath)
        
        params = {
            'RMS_shape': mat_data['RMS_shape'].flatten(),
            'RMS_scale': mat_data['RMS_scale'].flatten(),
            'slope_m': mat_data['slope_m'].flatten(),
            'slope_sd': mat_data['slope_sd'].flatten()
        }
        
        print(f"✓ Loaded artifact parameters from {filepath}")
        return params
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        print("Using default parameters instead.")
        return {
            'RMS_shape': np.array([2.0, 2.5, 3.0, 2.2]),
            'RMS_scale': np.array([0.5, 0.6, 0.7, 0.55]),
            'slope_m': np.array([-1.5, -2.0, -2.5, -1.8]),
            'slope_sd': np.array([0.3, 0.35, 0.4, 0.32])
        }


def load_rr_intervals(filepath, rhythm_type='SR', max_beats=None):
    """
    Load RR intervals from .mat data files
    
    Parameters:
    -----------
    filepath : str
        Path to RR interval .mat file (e.g., DATA_RR_SR_real.mat or DATA_RR_AF_real.mat)
    rhythm_type : str
        'SR' for sinus rhythm or 'AF' for atrial fibrillation
    max_beats : int, optional
        Maximum number of RR intervals to load
        
    Returns:
    --------
    array
        RR intervals in milliseconds
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"RR interval file not found: {filepath}")
    
    try:
        mat_data = sio.loadmat(filepath)
        
        # Try different possible variable names
        possible_names = ['rr', 'RR', 'rr_intervals', 'RR_intervals', 'data']
        rr = None
        
        for name in possible_names:
            if name in mat_data:
                rr = mat_data[name].flatten()
                break
        
        if rr is None:
            # If standard names not found, look for the first numeric array
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    rr = value.flatten()
                    break
        
        if rr is None:
            raise ValueError(f"Could not find RR interval data in {filepath}")
        
        if max_beats is not None and len(rr) > max_beats:
            # Randomly select a segment
            start_idx = np.random.randint(0, len(rr) - max_beats)
            rr = rr[start_idx:start_idx + max_beats]
        
        print(f"✓ Loaded {len(rr)} RR intervals from {filepath}")
        return rr
        
    except Exception as e:
        raise RuntimeError(f"Error loading RR intervals from {filepath}: {e}")


def generate_synthetic_rr(num_beats, rhythm_type='SR', mean_rr=800, std_rr=50):
    """
    Generate synthetic RR intervals (simplified version)
    
    Parameters:
    -----------
    num_beats : int
        Number of RR intervals to generate
    rhythm_type : str
        'SR' for sinus rhythm or 'AF' for atrial fibrillation
    mean_rr : float
        Mean RR interval in milliseconds
    std_rr : float
        Standard deviation of RR intervals in milliseconds
        
    Returns:
    --------
    array
        Synthetic RR intervals in milliseconds
    """
    if rhythm_type == 'SR':
        # Sinus rhythm: more regular with respiratory variation
        # Add low-frequency modulation
        t = np.arange(num_beats)
        respiratory_freq = 0.25  # ~15 breaths per minute
        modulation = 1 + 0.1 * np.sin(2 * np.pi * respiratory_freq * t / num_beats)
        rr = mean_rr * modulation + std_rr * np.random.randn(num_beats)
    else:  # AF
        # Atrial fibrillation: more irregular
        rr = mean_rr + std_rr * 2.5 * np.random.randn(num_beats)
    
    # Ensure physiologically reasonable values (300-2000 ms)
    rr = np.clip(rr, 300, 2000)
    
    return rr


def save_ppg_data(filepath, ppg, ppg_model, artifact, rr=None, 
                  ppg_peak_idx=None, ppg_peak_val=None, sampling_freq=1000):
    """
    Save PPG data to file in NumPy format
    
    Parameters:
    -----------
    filepath : str
        Output filepath (without extension, will add .npz)
    ppg : array
        Complete PPG signal (pulsatile + artifact)
    ppg_model : array
        Pulsatile component only
    artifact : array
        Artifact component only
    rr : array, optional
        RR intervals used
    ppg_peak_idx : array, optional
        Indices of PPG peaks
    ppg_peak_val : array, optional
        Values at PPG peaks
    sampling_freq : int
        Sampling frequency in Hz
    """
    save_dict = {
        'PPG': ppg,
        'PPGmodel': ppg_model,
        'artifact': artifact,
        'sampling_freq': sampling_freq
    }
    
    if rr is not None:
        save_dict['rr'] = rr
    if ppg_peak_idx is not None:
        save_dict['PPGpeakIdx'] = ppg_peak_idx
    if ppg_peak_val is not None:
        save_dict['PPGpeakVal'] = ppg_peak_val
    
    output_file = filepath if filepath.endswith('.npz') else f"{filepath}.npz"
    np.savez(output_file, **save_dict)
    print(f"✓ Saved PPG data to {output_file}")


if __name__ == "__main__":
    # Test loading artifact parameters
    print("Testing data loader...")
    params = load_artifact_params('artifact_param.mat')
    print(f"RMS_shape: {params['RMS_shape']}")
    print(f"RMS_scale: {params['RMS_scale']}")
    print(f"slope_m: {params['slope_m']}")
    print(f"slope_sd: {params['slope_sd']}")
    
    # Test synthetic RR generation
    rr_sr = generate_synthetic_rr(50, 'SR')
    print(f"\nGenerated SR RR intervals: mean={np.mean(rr_sr):.1f}ms, std={np.std(rr_sr):.1f}ms")
    
    rr_af = generate_synthetic_rr(50, 'AF')
    print(f"Generated AF RR intervals: mean={np.mean(rr_af):.1f}ms, std={np.std(rr_af):.1f}ms")
