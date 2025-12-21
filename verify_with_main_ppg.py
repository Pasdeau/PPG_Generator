import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import subprocess

# Ensure local modules are found
sys.path.append(os.getcwd())

from ml_training.model_factory import create_model
from ml_training.dataset import cwt_ricker # Reuse logic from dataset.py

# Constants
MODEL_PATH = "checkpoints_cwt_v3/best_model.pth" # v3.0 Path
DATA_FILE = "output/python_ppg_test_v3.npz"
DEVICE = "cpu"

# --- USER CONFIGURATION ---
TEST_PULSE_TYPE = 5         # 1-5
TEST_NUM_BEATS = 80         # Duration
TEST_ADD_ARTIFACTS = True   # True/False
TEST_ARTIFACT_TYPE = 2      # 1:Baseline 2:Forearm 3:Hand 4:HighFreq
TEST_ARTIFACT_INT = 3.0     # Interval (s)
TEST_ARTIFACT_DUR = 1.0     # Duration (s)
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
        # main_ppg.py defaults to output/something.npz, we need to rename or ensure checking right file
        # Check main_ppg arguments for output filename if possible, otherwise it auto-names
        # For simplicity, we assume main_ppg saves to 'output/python_ppg_test.npz' by default or similar
        # But we want to be safe. Let's just run it and find the latest file.
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("[+] main_ppg.py executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] main_ppg.py failed: {e}")
        sys.exit(1)

def get_latest_npz():
    # Find latest npz in output/
    files = [f for f in os.listdir("output") if f.endswith(".npz")]
    if not files: return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join("output", x)), reverse=True)
    return os.path.join("output", files[0])

def verify_on_generated_data():
    # 1. Generate Data
    run_main_ppg()
    
    # 2. Load Data
    data_path = get_latest_npz()
    if not data_path:
        print(f"[FAIL] No data found in output/.")
        return

    print(f"[-] Loading generated data from {data_path}...")
    data = np.load(data_path)
    ppg_signal = data['PPG'] # Combined Signal
    
    # Handle Artifact Label (might be 'artifact' or derived)
    if 'artifact' in data:
        artifact_signal = data['artifact']
    else:
        artifact_signal = np.zeros_like(ppg_signal)
    
    # 3. Preprocess for v3.0 (Length 8000, 34-Channels)
    target_len = 8000
    current_len = len(ppg_signal)
    
    print(f"[-] Signal Length: {current_len} (Target: {target_len})")
    
    # Pad/Crop
    if current_len < target_len:
        sig_input = np.pad(ppg_signal, (0, target_len - current_len))
        art_input = np.pad(artifact_signal, (0, target_len - current_len))
    else:
        sig_input = ppg_signal[:target_len]
        art_input = artifact_signal[:target_len]
        
    # --- Feature Engineering (v3.0) ---
    print("[-] Computing CWT (32 scales)...")
    
    # Ch 0: Amplitude (Norm)
    sig_amp = (sig_input - np.mean(sig_input)) / (np.std(sig_input) + 1e-6)
    
    # Ch 1: Velocity (Norm)
    velocity = np.diff(sig_input, prepend=sig_input[0])
    sig_vel = (velocity - np.mean(velocity)) / (np.std(velocity) + 1e-6)
    
    # Ch 2-33: CWT (Scales 1-32)
    scales = np.arange(1, 33)
    cwt_matrix = cwt_ricker(sig_input, scales)
    # Global Norm for CWT
    cwt_norm = (cwt_matrix - np.mean(cwt_matrix)) / (np.std(cwt_matrix) + 1e-6)
    
    # Stack: [34, 8000]
    tensor_input = np.vstack([
        sig_amp[np.newaxis, :],
        sig_vel[np.newaxis, :],
        cwt_norm
    ])
    tensor_in = torch.from_numpy(tensor_input).float().unsqueeze(0) # [1, 34, 8000]
    
    # 4. Load Model (v3.0)
    print(f"[-] Loading V3.0 Model (CWT)... Path: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] checkpoint not found locally at {MODEL_PATH}")
        print("Please download it from the remote server: ")
        print(f"Run: scp front.convergence.lip6.fr:~/checkpoints_cwt_v3/best_model.pth {MODEL_PATH}")
        # Create Dummy Model for Demo if missing? No, user wants to verify.
        return
        
    try:
        model = create_model('unet', input_length=8000, n_classes_seg=5, n_classes_clf=5, in_channels=34)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # 5. Inference
    print("[-] Running Inference...")
    with torch.no_grad():
        logits_clf, logits_seg = model(tensor_in)
        clf_probs = torch.softmax(logits_clf, dim=1).squeeze().numpy()
        pred_class = np.argmax(clf_probs)
        pred_mask = torch.argmax(logits_seg, dim=1).squeeze().numpy()
        
    # 6. Visualization
    print("[-] Visualizing...")
    classes = ['Sinus (N)', 'Premature (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
    
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.5])
    
    # Subplot 1: Signal + Detection
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(sig_input, 'k-', linewidth=0.8, label='PPG (Amp)')
    # Overlay noise
    mask_indices = np.where(pred_mask > 0)[0]
    if len(mask_indices) > 0:
        y_min, y_max = np.min(sig_input), np.max(sig_input)
        is_noise = (pred_mask > 0).astype(float)
        ax1.fill_between(np.arange(target_len), y_min, y_max, where=is_noise>0.5, 
                        color='red', alpha=0.3, label='AI Noise Detected')
    ax1.set_title(f"Model Prediction (Type: {classes[pred_class]})")
    ax1.legend()
    
    # Subplot 2: Ground Truth vs Mask Class
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(art_input, 'g-', linewidth=0.8, alpha=0.6, label='Artifact (Ground Truth)')
    ax2.set_ylabel("Artifact Amp")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(pred_mask, 'r-', linewidth=1.5, label='AI Mask Class (0-4)')
    ax2_twin.set_ylabel("Class ID")
    ax2_twin.set_ylim(-0.5, 4.5)
    ax2_twin.set_yticks([0, 1, 2, 3, 4])
    ax2_twin.set_yticklabels(['Clean', 'Disp', 'Forearm', 'Hand', 'Poor'])
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    # Subplot 3: CWT Spectrogram (This is what the model sees!)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    # Plot heatmap of cwt_norm
    im = ax3.imshow(cwt_norm, aspect='auto', cmap='jet', extent=[0, target_len, 32, 1], origin='upper')
    ax3.set_title("CWT Scalogram (Model Input Channels 2-33)")
    ax3.set_ylabel("Scale (Low=High Freq)")
    ax3.set_xlabel("Samples")
    plt.colorbar(im, ax=ax3, label='Energy (Norm)')
    
    plt.tight_layout()
    os.makedirs("validation_v3", exist_ok=True)
    out_path = f"validation_v3/verify_P{TEST_PULSE_TYPE}_Arts_{TEST_ADD_ARTIFACTS}.png"
    plt.savefig(out_path, dpi=150)
    print(f"[+] Saved visualization to {out_path}")
    print(f"\n[SUMMARY] Detected: {classes[pred_class]} | Noise: {'Yes' if np.any(pred_mask>0) else 'No'}")

if __name__ == "__main__":
    verify_on_generated_data()
