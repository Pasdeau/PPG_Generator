import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
from pathlib import Path
from ml_training.model_factory import create_model
from ml_training.dataset import PPGDataset, preprocess_signal

# Label Mappings
PULSE_LABELS = {0: 'Sinus (N)', 1: 'Premature (S)', 2: 'Ventricular (V)', 3: 'Fusion (F)', 4: 'Unknown (Q)'}
# Artifact mapping might need adjustment based on how dataset stores it
# But dataset returns class 0-4 for artifacts if task='artifact'
# We need to map labels back to names.
ARTIFACT_NAMES = ['Clean', 'Device', 'Forearm', 'Hand', 'PoorContact']

def verify_on_test_set(model_path, data_dir, output_dir, num_samples=50):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    print("Loading model...")
    model = create_model('dual', in_channels=34, n_classes_seg=5, n_classes_clf=5, attention=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load Test Dataset
    # Structure is flat: data_dir/signals/*.npz
    print("Loading test samples...")
    test_files = glob.glob(os.path.join(data_dir, 'signals', '*.npz'))
    if not test_files:
        test_files = glob.glob(os.path.join(data_dir, '*.npz'))
    
    selected_files = np.random.choice(test_files, min(num_samples, len(test_files)), replace=False)
    
    results = []
    
    print(f"Processing {len(selected_files)} samples...")
    
    for i, npz_path in enumerate(selected_files):
        # Load Data
        data = np.load(npz_path)
        signal = data['PPG'] if 'PPG' in data else data['signal']
        
        # Load Metadata
        # Try finding json
        json_path = npz_path.replace('signals', 'labels').replace('.npz', '.json')
        if not os.path.exists(json_path):
             json_path = npz_path.replace('.npz', '_meta.json')
        
        gt_pulse_label = "?"
        gt_noise_label = "?"
        gt_mask = None
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                meta = json.load(f)
                if 'metadata' in meta: meta.update(meta.pop('metadata')) # flattening
                
                p_type = meta.get('pulse_type', 0)
                gt_pulse_label = PULSE_LABELS.get(p_type - 1 if p_type > 0 else 0, str(p_type)) # Adjust 1-based to 0-based
                gt_pulse_idx = p_type - 1 if p_type > 0 else 0
                
                a_type = meta.get('artifact_type', 'clean')
                gt_noise_label = a_type
                
                if 'artifact_timeline' in meta:
                    gt_mask = np.array(meta['artifact_timeline'])
        
        # Preprocess
        tensor = preprocess_signal(signal).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            pred_clf, pred_seg = model(tensor)
            
        pred_p_idx = pred_clf.argmax(1).item()
        pred_mask_np = pred_seg.argmax(1).squeeze().cpu().numpy()
        conf = torch.softmax(pred_clf, dim=1).max().item()
        
        # Visualize
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        
        t = np.arange(len(signal)) / 1000
        axes[0].plot(t, signal, 'k-', linewidth=0.5)
        axes[0].set_title(f"Sample {i} | GT: Pulse={gt_pulse_label}, Noise={gt_noise_label}")
        
        if gt_mask is not None:
             # Resize mask if needed
             if len(gt_mask) != len(t):
                 gt_mask = np.resize(gt_mask, len(t))
             axes[1].fill_between(t, 0, gt_mask, alpha=0.3, color='green', label='GT Mask')
             
        axes[1].fill_between(t, 0, pred_mask_np, alpha=0.3, color='red', label='Pred Mask')
        axes[1].set_ylim(-0.5, 5)
        axes[1].legend()
        axes[1].set_title(f"Pred: Pulse={PULSE_LABELS.get(pred_p_idx, 'Unk')} ({conf:.1%})")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"remote_verif_{i}.png"))
        plt.close()
        
        # Tracking
        pulse_match = (pred_p_idx == gt_pulse_idx)
        results.append(pulse_match)
        print(f"Sample {i}: Pulse Match={pulse_match} ({gt_pulse_label} vs {PULSE_LABELS.get(pred_p_idx)})")

    acc = sum(results) / len(results)
    print(f"Test Accuracy (Subset): {acc*100:.2f}%")

if __name__ == '__main__':
    verify_on_test_set(
        'checkpoints_dual_v4/best_model.pth',
        '/home/wenwang/ppg_training_data_v3',
        'remote_validation_results'
    )
