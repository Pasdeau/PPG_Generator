import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure local modules are found
sys.path.append(os.getcwd())

from ml_training.model_factory import create_model
from generate_segmentation_data import generate_segmentation_sample

MODEL_PATH = "local_best_model_v2.pth"
DEVICE = "cpu"

def visualize_inference():
    print("[-] Loading model...")
    model = create_model('unet', input_length=8000, n_classes_seg=5, n_classes_clf=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Generate a sample with artifacts
    print("[-] Generating sample with artifacts...")
    sig, gt_mask, gt_label = generate_segmentation_sample(sample_id=np.random.randint(1000), Fd=1000)
    
    # Normalize
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
    tensor_in = torch.from_numpy(sig_norm).float().view(1, 1, -1)
    
    # Inference
    print("[-] Running inference...")
    with torch.no_grad():
        logits_clf, logits_seg = model(tensor_in)
        
        # Classification
        clf_probs = torch.softmax(logits_clf, dim=1).squeeze().numpy()
        pred_label = np.argmax(clf_probs)
        
        # Segmentation
        pred_mask = torch.argmax(logits_seg, dim=1).squeeze().numpy()
        
    # --- Plotting ---
    print("[-] Creating visualization dashboard...")
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Main Signal + Segmentation Overlay
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(sig, color='black', linewidth=0.8, label='PPG Signal')
    
    # Highlight detected artifacts
    # Create a red overlay where mask > 0
    mask_indices = np.where(pred_mask > 0)[0]
    if len(mask_indices) > 0:
        # Split into contiguous segments for cleaner plotting
        segments = np.split(mask_indices, np.where(np.diff(mask_indices) != 1)[0]+1)
        for i, seg in enumerate(segments):
            label = "Detected Noise" if i == 0 else None
            ax1.axvspan(seg[0], seg[-1], color='red', alpha=0.3, label=label)
            
    ax1.set_title(f"Input Signal & Detected Artifacts (IoU Check)", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_xlim(0, 8000)
    
    # 2. Ground Truth vs Pred Mask (Strip Chart)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.imshow(gt_mask[np.newaxis, :], aspect='auto', cmap='Greens', extent=[0, 8000, 0, 1], alpha=0.8)
    ax2.text(100, 0.5, "Ground Truth", color='black', fontsize=10, ha='left', va='center', backgroundcolor='white')
    
    ax2.imshow(pred_mask[np.newaxis, :], aspect='auto', cmap='Reds', extent=[0, 8000, -1.2, -0.2], alpha=0.8)
    ax2.text(100, -0.7, "Prediction", color='black', fontsize=10, ha='left', va='center', backgroundcolor='white')
    
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks([])
    ax2.set_title("Segmentation Comparison (Green=GT, Red=Pred)", fontsize=10)
    ax2.set_xlim(0, 8000)
    
    # 3. Classification Probabilities (Bar Chart)
    ax3 = fig.add_subplot(gs[2, 0])
    classes = ['Sinus (N)', 'Premature (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
    colors = ['gray'] * 5
    colors[gt_label] = 'green' # GT
    if pred_label != gt_label:
        colors[pred_label] = 'red' # Wrong Pred
    else:
        colors[pred_label] = 'blue' # Correct Pred (overwrites GT green if same)
        
    bars = ax3.bar(classes, clf_probs, color=colors, alpha=0.7)
    ax3.set_ylim(0, 1.1)
    ax3.set_title(f"Waveform Classification (GT: {classes[gt_label]})", fontsize=10)
    ax3.set_ylabel("Confidence")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=8)

    # 4. Text Summary
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    summary_text = (
        f"Model Diagnosis Summary:\n"
        f"------------------------\n"
        f"Predicted Class: {classes[pred_label]} ({clf_probs[pred_label]:.1%})\n"
        f"True Class:      {classes[gt_label]}\n\n"
        f"Noise Detected:  {np.sum(pred_mask > 0)} samples\n"
        f"Result:          {'CORRECT' if pred_label == gt_label else 'INCORRECT'}"
    )
    ax4.text(0.1, 0.5, summary_text, fontsize=12, fontfamily='monospace', va='center')
    
    plt.tight_layout()
    os.makedirs("validation", exist_ok=True)
    plt.savefig("validation/model_visualization.png", dpi=150)
    print(f"[+] Dashboard saved to: {os.getcwd()}/validation/model_visualization.png")

if __name__ == "__main__":
    visualize_inference()
