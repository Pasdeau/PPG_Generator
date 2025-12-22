import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Ensure local modules are found
sys.path.append(os.getcwd())

from ml_training.model_factory import create_model
from ml_training.dataset import create_dataloaders

# Configuration
MODEL_PATH = "checkpoints_cwt_v3/best_model.pth"
DATASET_PATH = os.path.expanduser("~/ppg_training_data_v3") # Remote path default
# If running locally without full dataset, this might fail. 
# We'll check if local 'output' directory works or fallback to user-provided path.
DEVICE = "cpu"

def visualize_batch(data_dir, num_samples=5):
    print(f"[-] Loading data from {data_dir}...")
    
    # Check if dir exists
    if not os.path.exists(data_dir):
        print(f"[ERROR] Dataset directory not found: {data_dir}")
        print("Tip: If you don't have the full dataset locally, run 'main_ppg.py' to generate some samples in 'output/signals' and point to that.")
        return

    # Create simplified loader (batch_size = num_samples)
    # We use 'multitask' to get (signal, mask, label)
    # Note: create_dataloaders expects 'data_dir' to contain 'signals/*.npz'
    # If the user points to 'output', we need to make sure structure matches or Dataset class handles it.
    # Dataset class recursively searches .npz, so pointing to 'output' should work!
    
    try:
        train_loader, val_loader, _ = create_dataloaders(
            data_dir, 
            task='multitask', 
            batch_size=num_samples, 
            num_workers=0, # Main thread for debug
            augment=False
        )
    except Exception as e:
        print(f"[FAIL] Failed to create dataloader: {e}")
        return

    # Get one batch from val_loader
    print("[-] Fetching a batch...")
    data_iter = iter(val_loader)
    try:
        signals, masks, labels = next(data_iter)
    except StopIteration:
        print("[WARN] Validation loader empty, trying training loader...")
        try:
            data_iter = iter(train_loader)
            signals, masks, labels = next(data_iter)
        except StopIteration:
            print("[FAIL] All loaders empty.")
            return
        
    # Load Model
    print(f"[-] Loading V3.0 Model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] detailed model not found at {MODEL_PATH}")
        print("Please download it: scp front.convergence.lip6.fr:~/checkpoints_cwt_v3/best_model.pth checkpoints_cwt_v3/")
        return

    model = create_model('unet', input_length=8000, n_classes_seg=5, n_classes_clf=5, in_channels=34)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"[FAIL] Model load failed: {e}")
        return
        
    model.eval()
    
    # Inference
    print("[-] Running inference...")
    with torch.no_grad():
        logits_clf, logits_seg = model(signals)
        
        preds_clf = torch.argmax(torch.softmax(logits_clf, dim=1), dim=1).numpy()
        preds_mask = torch.argmax(logits_seg, dim=1).numpy()
        
    # Visualization
    print("[-] Plotting batch...")
    signals_np = signals.numpy()
    masks_np = masks.numpy()
    labels_np = labels.numpy()
    
    # Plot N samples stacked vertically
    actual_batch_size = len(signals_np)
    fig, axes = plt.subplots(actual_batch_size, 1, figsize=(12, 3*actual_batch_size), sharex=True)
    if actual_batch_size == 1: axes = [axes]
    
    classes = ['Sinus (N)', 'Premature (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
    
    for i in range(actual_batch_size):
        # Determine class colors
        gt_cls = labels_np[i]
        pred_cls = preds_clf[i]
        color_cls = 'green' if gt_cls == pred_cls else 'red'
        
        # Plot Signal (Channel 0 is Amp)
        sig = signals_np[i, 0, :]
        axes[i].plot(sig, 'k-', linewidth=0.6, label='Signal')
        
        # Plot GT Mask (Green Bars at bottom)
        gt_noise = (masks_np[i] > 0).astype(float)
        # axes[i].fill_between(np.arange(8000), -3, -2.5, where=gt_noise>0.5, color='green', alpha=0.5, label='GT Artifact')
        
        # Plot Pred Mask (Red Overlay)
        pred_noise = (preds_mask[i] > 0).astype(float)
        y_min, y_max = sig.min(), sig.max()
        axes[i].fill_between(np.arange(8000), y_min, y_max, where=pred_noise>0.5, color='red', alpha=0.3, label='Pred Artifact')
        
        title = f"Sample {i+1} | GT: {classes[gt_cls]} | Pred: {classes[pred_cls]}"
        axes[i].set_title(title, color=color_cls, fontweight='bold')
        
        if i == 0: axes[i].legend(loc='upper right')
        
    os.makedirs("validation_batch", exist_ok=True)
    out_path = "validation_batch/v3_batch_results.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[+] Batch result saved to {out_path}")

if __name__ == "__main__":
    # Check for argument path
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "output"
    visualize_batch(data_path, num_samples=5)
