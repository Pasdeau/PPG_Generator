import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import subprocess

# Ensure local modules are found
sys.path.append(os.getcwd())

from ml_training.model_factory import create_model

# Constants
MODEL_PATH = "checkpoints_seg_v2/best_model.pth" # Default output path for v2.4
DATA_FILE = "output/python_ppg_test.npz"
DEVICE = "cpu"

# --- USER CONFIGURATION ---
TEST_PULSE_TYPE = 5         # 1-5
TEST_NUM_BEATS = 80         # Duration (approx 1 beat = 0.8s)
TEST_ADD_ARTIFACTS = True   # True/False
TEST_ARTIFACT_TYPE = 2      # 1:Baseline 2:Motion(Forearm) 3:Motion(Hand) 4:HighFreq
TEST_ARTIFACT_INT = 3.0     # Interval between artifacts (seconds)
TEST_ARTIFACT_DUR = 1.0     # Duration of each artifact (seconds)
# --------------------------

def run_main_ppg():
    print("[-] Running main_ppg.py to generate fresh data...")
    cmd = [
        sys.executable, "main_ppg.py",
        "--num_beats", str(TEST_NUM_BEATS),
        "--pulse_type", str(TEST_PULSE_TYPE),
        "--dur_mu0", str(TEST_ARTIFACT_INT),
        "--dur_mu", str(TEST_ARTIFACT_DUR),
        "--artifact_type", str(TEST_ARTIFACT_TYPE)
    ]
    if not TEST_ADD_ARTIFACTS:
        cmd.append("--no_artifacts")
        
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("[+] main_ppg.py executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] main_ppg.py failed: {e}")
        sys.exit(1)

def verify_on_generated_data():
    # 1. Generate Data
    run_main_ppg()
    
    # 2. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"[FAIL] Data file {DATA_FILE} not found.")
        return

    print(f"[-] Loading generated data from {DATA_FILE}...")
    data = np.load(DATA_FILE)
    ppg_signal = data['PPG'] # Combined Signal
    artifact_signal = data['artifact'] # True Artifact (Continuous)
    # Note: keys in save_ppg_data: 'ppg_combined', 'ppg_pulsatile', 'artifact', 'rr', etc.
    
    # 3. Preprocess for Model (Length 8000, Normalized)
    target_len = 8000
    current_len = len(ppg_signal)
    
    print(f"[-] Signal Length: {current_len} (Target: {target_len})")
    
    if current_len < target_len:
        # Pad
        sig_input = np.pad(ppg_signal, (0, target_len - current_len))
        art_input = np.pad(artifact_signal, (0, target_len - current_len))
    else:
        # Crop (Center crop or specific crop? Start is usually cleaner)
        sig_input = ppg_signal[:target_len]
        art_input = artifact_signal[:target_len]
        
    # Standardize Channel 1: Amplitude
    sig_amp = (sig_input - np.mean(sig_input)) / (np.std(sig_input) + 1e-6)
    
    # Generate Channel 2: Velocity (Gradient)
    sig_vel = np.gradient(sig_input)
    # Standardize Channel 2 independently
    sig_vel = (sig_vel - np.mean(sig_vel)) / (np.std(sig_vel) + 1e-6)
    
    # Stack: [1, 2, 8000]
    tensor_input = np.stack([sig_amp, sig_vel], axis=0)
    tensor_in = torch.from_numpy(tensor_input).float().unsqueeze(0) # Add batch dim
    
    # 4. Load Model
    print("[-] Loading V2.4 Model (Dual-Channel)...")
    try:
        model = create_model('unet', input_length=8000, n_classes_seg=5, n_classes_clf=5, in_channels=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except FileNotFoundError:
        print(f"[ERROR] Model file not found at {MODEL_PATH}")
        print("Please wait for training to complete and download the checkpoint.")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # 5. Inference
    print("[-] Running Inference...")
    with torch.no_grad():
        logits_clf, logits_seg = model(tensor_in)
        
        # Classification
        clf_probs = torch.softmax(logits_clf, dim=1).squeeze().numpy()
        pred_class = np.argmax(clf_probs)
        
        # Segmentation
        pred_mask = torch.argmax(logits_seg, dim=1).squeeze().numpy()
        
    # 6. Visualization
    print("[-] Visualizing comparison...")
    
    classes = ['Type 1: Sinus (N)', 'Type 2: Premature (S)', 'Type 3: Ventricular (V)', 'Type 4: Fusion (F)', 'Type 5: Unknown (Q)']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Signal + Model Detection
    ax1.plot(sig_input, 'k-', linewidth=0.8, label='PPG (Amplitude)')
    # Optional: Plot gradient in background? Maybe too messy.
    
    # Overlay Prediction (Red)
    mask_indices = np.where(pred_mask > 0)[0]
    if len(mask_indices) > 0:
        # Simple fill for visualization
        # We can create a boolean curve for fill_between
        is_noise = (pred_mask > 0).astype(float)
        # Scale for visibility
        y_min, y_max = np.min(sig_input), np.max(sig_input)
        ax1.fill_between(np.arange(target_len), y_min, y_max, where=is_noise>0.5, 
                        color='red', alpha=0.3, label='AI Detected Noise')
    
    ax1.set_title(f"Model Prediction (Class: {classes[pred_class]}, Conf: {clf_probs[pred_class]:.1%})", fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylabel("Amplitude")
    
    # Plot 2: Artifact Ground Truth (from main_ppg) vs Model Mask
    # main_ppg artifact is an amplitude signal. We plot it.
    ax2.plot(art_input, 'g-', linewidth=0.8, label='Injected Artifact (Truth)')
    ax2.set_ylabel("Artifact Amp")
    
    # Overlay Model Mask on secondary axis or just as bars
    ax2_twin = ax2.twinx()
    ax2_twin.plot(pred_mask, 'r--', linewidth=1.0, alpha=0.6, label='AI Mask (0-4)')
    ax2_twin.set_ylabel("AI Mask Class", color='red')
    ax2_twin.set_ylim(-0.5, 4.5)
    ax2_twin.set_yticks([0, 1, 2, 3, 4])
    
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    ax2.set_title("Ground Truth Artifact vs AI Segmentation Mask")
    ax2.set_xlabel("Samples")
    
    plt.tight_layout()
    os.makedirs("validation_v2_4", exist_ok=True)
    if TEST_ADD_ARTIFACTS:
        out_path = f"validation_v2_4/verify_Pulse_Type{TEST_PULSE_TYPE}_Noise_Type{TEST_ARTIFACT_TYPE}.png"
    else:
        out_path = f"validation_v2_4/verify_Pulse_Type{TEST_PULSE_TYPE}_Clean.png"
    plt.savefig(out_path, dpi=150)
    print(f"[+] Result saved to {out_path}")
    
    # Explicit Report for User
    print("\n" + "="*40)
    print("     MODEL DIAGNOSIS REPORT (v2.4)")
    print("="*40)
    print(f"DETECTED WAVEFORM: Type {classes[pred_class]}")
    print(f"CONFIDENCE:        {clf_probs[pred_class]:.1%}")
    print(f"NOISE STATUS:      {'Signal contains noise' if np.any(pred_mask > 0) else 'Signal is clean'}")
    print("="*40 + "\n")
    
    # Qualitative Check
    if np.any(pred_mask > 0) and np.std(art_input) > 0.1:
        print("[SUCCESS] Model detected noise in the noisy signal.")
    elif not np.any(pred_mask > 0) and np.std(art_input) < 0.01:
        print("[SUCCESS] Model correctly identified clean signal.")
    else:
        print("[WARN] Detection might vary. Check the plot.")

if __name__ == "__main__":
    verify_on_generated_data()
