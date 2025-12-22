import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import subprocess
import argparse

# Ensure local modules are found
sys.path.append(os.getcwd())

from ml_training.model_factory import create_model
from ml_training.dataset import cwt_ricker # Reuse logic from dataset.py

# Constants (Defaults)
DEFAULT_MODEL_PATH = "checkpoints_cwt_v3/best_model.pth" 
DATA_FILE = "output/python_ppg_test_v3.npz"
DEVICE = "cpu"

def run_main_ppg(pulse_type, artifact_type, add_artifacts):
    print("[-] Running main_ppg.py to generate fresh data...")
    cmd = [
        sys.executable, "main_ppg.py",
        "--num_beats", "80",
        "--pulse_type", str(pulse_type),
        "--dur_mu0", "3.0",
        "--dur_mu", "1.0",
        "--artifact_type", str(artifact_type)
    ]
    if not add_artifacts:
        cmd.append("--no_artifacts")
        
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] main_ppg.py failed: {e}")
        sys.exit(1)

def get_latest_npz():
    files = [f for f in os.listdir("output") if f.endswith(".npz")]
    if not files: return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join("output", x)), reverse=True)
    return os.path.join("output", files[0])

def verify_on_generated_data(pulse_type=3, artifact_type=2, add_artifacts=True, 
                           model_path=DEFAULT_MODEL_PATH, output_dir="validation_v3", use_attention=False):
    # 1. Generate Data
    run_main_ppg(pulse_type, artifact_type, add_artifacts)
    
    # 2. Load Data
    data_path = get_latest_npz()
    if not data_path: return

    print(f"[-] Loading generated data from {data_path}...")
    data = np.load(data_path)
    ppg_signal = data['PPG']
    if 'artifact' in data: artifact_signal = data['artifact']
    else: artifact_signal = np.zeros_like(ppg_signal)
    
    # 3. Preprocess
    target_len = 8000
    current_len = len(ppg_signal)
    
    if current_len < target_len:
        sig_input = np.pad(ppg_signal, (0, target_len - current_len))
        art_input = np.pad(artifact_signal, (0, target_len - current_len))
    else:
        sig_input = ppg_signal[:target_len]
        art_input = artifact_signal[:target_len]
        
    # Feature Engineering
    sig_amp = (sig_input - np.mean(sig_input)) / (np.std(sig_input) + 1e-6)
    velocity = np.diff(sig_input, prepend=sig_input[0])
    sig_vel = (velocity - np.mean(velocity)) / (np.std(velocity) + 1e-6)
    
    scales = np.arange(1, 33)
    cwt_matrix = cwt_ricker(sig_input, scales)
    cwt_norm = (cwt_matrix - np.mean(cwt_matrix)) / (np.std(cwt_matrix) + 1e-6)
    
    tensor_input = np.vstack([sig_amp[np.newaxis, :], sig_vel[np.newaxis, :], cwt_norm])
    tensor_in = torch.from_numpy(tensor_input).float().unsqueeze(0)
    
    # 4. Load Model
    print(f"[-] Loading Model from {model_path} (Attention={use_attention})...")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return
        
    try:
        model = create_model('unet', input_length=8000, n_classes_seg=5, n_classes_clf=5, 
                           in_channels=34, attention=use_attention)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # 5. Inference
    with torch.no_grad():
        logits_clf, logits_seg = model(tensor_in)
        clf_probs = torch.softmax(logits_clf, dim=1).squeeze().numpy()
        pred_class = np.argmax(clf_probs)
        pred_mask = torch.argmax(logits_seg, dim=1).squeeze().numpy()
        
    # 6. Visualization
    classes = ['Sinus (N)', 'Premature (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
    noise_classes = ['Clean', 'Baseline', 'Forearm', 'Hand', 'HighFreq']
    
    # Determine dominant noise class
    unique, counts = np.unique(pred_mask, return_counts=True)
    noise_counts = dict(zip(unique, counts))
    print(f"[DEBUG] Mask Counts (0=Clean, 1=Base, 2=Fore, 3=Hand, 4=High): {noise_counts}")
    
    # Exclude 0 (Clean) to find dominant noise
    non_clean_counts = {k: v for k, v in noise_counts.items() if k > 0}
    
    if non_clean_counts:
        dom_noise_idx = max(non_clean_counts, key=non_clean_counts.get)
        noise_str = f"{noise_classes[dom_noise_idx]} (Type {dom_noise_idx})"
    else:
        noise_str = "None (Clean)"
        
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.5])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(sig_input, 'k-', linewidth=0.8, label='PPG')
    mask_indices = np.where(pred_mask > 0)[0]
    if len(mask_indices) > 0:
        y_min, y_max = np.min(sig_input), np.max(sig_input)
        is_noise = (pred_mask > 0).astype(float)
        ax1.fill_between(np.arange(target_len), y_min, y_max, where=is_noise>0.5, 
                        color='red', alpha=0.3, label='Noise Detected')
    
    plot_title = f"Pred: Pulse={classes[pred_class]} ({clf_probs[pred_class]*100:.1f}%) | Noise={noise_str}"
    ax1.set_title(plot_title, fontsize=12, fontweight='bold')
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(art_input, 'g-', linewidth=0.8, alpha=0.6, label='Artifact GT')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(pred_mask, 'r-', linewidth=1.5, label='Pred Mask')
    ax2_twin.set_ylim(-0.5, 4.5)
    ax2_twin.set_yticks([0, 1, 2, 3, 4])
    ax2_twin.set_yticklabels(['Clean', 'Base', 'Forearm', 'Hand', 'HighFreq'])
    ax2.legend(loc='upper left'); ax2_twin.legend(loc='upper right')
    
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    im = ax3.imshow(cwt_norm, aspect='auto', cmap='jet', extent=[0, target_len, 32, 1], origin='upper')
    plt.colorbar(im, ax=ax3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/verify_P{pulse_type}_N{artifact_type}.png"
    plt.savefig(out_path, dpi=150)
    print(f"[+] Saved {out_path}")
    print(f"[RESULT] Pulse: {classes[pred_class]} | Noise: {'Yes' if np.any(pred_mask>0) else 'No'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pulse_type', type=int, default=3)
    parser.add_argument('--artifact_type', type=int, default=2)
    parser.add_argument('--no_artifacts', action='store_true')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--output_dir', type=str, default="validation_v3")
    parser.add_argument('--attention', action='store_true', help="Use SE-Attention model structure")
    args = parser.parse_args()
    
    verify_on_generated_data(
        pulse_type=args.pulse_type,
        artifact_type=args.artifact_type,
        add_artifacts=not args.no_artifacts,
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_attention=args.attention
    )
