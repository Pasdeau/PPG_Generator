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

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def test_level_1_loading():
    print_header("LEVEL 1: Model Loading & Basic Forward Pass")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[FAIL] Model file {MODEL_PATH} not found!")
        return False

    try:
        print(f"[-] Loading model from {MODEL_PATH}...")
        model = create_model('unet', input_length=8000, n_classes_seg=5, n_classes_clf=5)
        # Handle state dict loading carefully
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        print("[+] Model loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Error loading model: {e}")
        return False

    # Dummy Input
    print("[-] Running dummy forward pass (Input: [1, 1, 8000])...")
    dummy_input = torch.randn(1, 1, 8000)
    try:
        with torch.no_grad():
            logits_clf, logits_seg = model(dummy_input)
        
        print(f"[+] Output Shapes -> Clf: {logits_clf.shape}, Seg: {logits_seg.shape}")
        
        if logits_clf.shape == (1, 5) and logits_seg.shape == (1, 5, 8000):
            print("[SUCCESS] Shape check passed.")
            return model
        else:
            print(f"[FAIL] Unexpected output shapes.")
            return False
    except Exception as e:
        print(f"[FAIL] Forward pass error: {e}")
        return False

def test_level_2_clean(model):
    print_header("LEVEL 2: Clean Signal Inference (PULSE TYPE)")
    
    # Generate clean signal
    print("[-] Generating synthetic signal...")
    # generate_segmentation_sample returns: signal, mask, class_label (0-indexed)
    sig, mask, label = generate_segmentation_sample(sample_id=101, Fd=1000)
    
    # Normalize
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
    tensor_in = torch.from_numpy(sig_norm).float().view(1, 1, -1)
    
    print(f"[-] Ground Truth Label: Type {label+1}")
    
    with torch.no_grad():
        l_clf, l_seg = model(tensor_in)
        pred_cls = torch.argmax(l_clf, dim=1).item()
        conf = torch.softmax(l_clf, dim=1)[0, pred_cls].item()
    
    print(f"[-] Predicted Label:    Type {pred_cls+1} (Conf: {conf:.2%})")
    
    if pred_cls == label:
        print("[SUCCESS] Classification correct!")
    else:
        print(f"[WARN] Classification mismatch (Expected {label+1}, Got {pred_cls+1}).")
    return True

def test_level_3_noisy(model):
    print_header("LEVEL 3: Noisy Signal Inference (SEGMENTATION)")
    
    print("[-] Generating NOISY signal with artifacts...")
    # Sample ID ensures some randomness, generator logic handles noise injection
    sig, mask, label = generate_segmentation_sample(sample_id=999, Fd=1000)
    
    # Check artifacts
    has_artifacts = np.any(mask > 0)
    unique, counts = np.unique(mask, return_counts=True)
    print(f"[-] Input contains artifacts? {has_artifacts}")
    if has_artifacts:
        print(f"    Artifact summary: {dict(zip(unique, counts))}")

    # Normalize
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
    tensor_in = torch.from_numpy(sig_norm).float().view(1, 1, -1)
    
    with torch.no_grad():
        _, l_seg = model(tensor_in)
        pred_mask = torch.argmax(l_seg, dim=1).squeeze().numpy()
        
    # IoU
    overlap = np.sum((mask > 0) & (pred_mask > 0))
    union = np.sum((mask > 0) | (pred_mask > 0))
    iou = overlap / (union + 1e-6)
    
    print(f"[-] Predicted Artifacts: {np.sum(pred_mask > 0)} samples")
    print(f"[-] IoU (Artifact Detection): {iou:.4f}")
    
    # Save to validation folder
    print("[-] Saving plot...")
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(sig)
    plt.title("Input Signal (Normalized)")
    plt.subplot(3, 1, 2)
    plt.plot(mask, color='green', label='GT Mask')
    plt.title("Ground Truth Mask")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(pred_mask, color='red', label='Pred Mask')
    plt.title(f"Predicted Mask (IoU={iou:.2f})")
    plt.legend()
    plt.tight_layout()
    
    os.makedirs("validation", exist_ok=True)
    out_path = "validation/test_v2_result.png"
    plt.savefig(out_path)
    print(f"[+] Visualization saved to '{out_path}'")
    
    if iou > 0.4 or (not has_artifacts and np.sum(pred_mask>0) < 200):
        print("[SUCCESS] Segmentation looks reasonable.")
    elif has_artifacts:
        print("[WARN] Segmentation IoU low.")
    
    return True

if __name__ == "__main__":
    model = test_level_1_loading()
    if model:
        test_level_2_clean(model)
        test_level_3_noisy(model)
        print("\n[ALL TESTS COMPLETED]")
