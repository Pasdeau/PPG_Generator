import matplotlib.pyplot as plt
import re
import os
import sys
import argparse

def parse_and_plot(log_file='training_log_dump.txt', out_path='validation/training_curves_v3.png'):
    if not os.path.exists(log_file):
        print(f"[ERROR] Log file not found: {log_file}")
        return

    train_loss, val_loss = [], []
    train_clf_acc, val_clf_acc = [], []
    train_seg_acc, val_seg_acc = [], [] # New for v3.0
    
    print(f"[-] Parsing log file: {log_file}...")
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # Regex for v3.0 format:
        # [Train] Loss: 2.8784 | Classifier Acc: 19.96% | Seg Acc: 97.40%
        loss_match = re.search(r'Loss:\s*([\d\.]+)', line)
        clf_acc_match = re.search(r'Classifier Acc:\s*([\d\.]+)%', line)
        seg_acc_match = re.search(r'Seg Acc:\s*([\d\.]+)%', line)
        
        if loss_match and clf_acc_match:
            loss = float(loss_match.group(1))
            clf_acc = float(clf_acc_match.group(1))
            # Handle optional Seg Acc (backward compatibility)
            seg_acc = float(seg_acc_match.group(1)) if seg_acc_match else 0.0
            
            if '[Train]' in line:
                train_loss.append(loss)
                train_clf_acc.append(clf_acc)
                if seg_acc_match: train_seg_acc.append(seg_acc)
            elif '[Val]' in line:
                val_loss.append(loss)
                val_clf_acc.append(clf_acc)
                if seg_acc_match: val_seg_acc.append(seg_acc)
                
    if not train_loss:
        print("[WARN] No metrics found in log file.")
        return

    # Plotting
    epochs = range(1, len(train_loss) + 1)
    min_len = min(len(train_loss), len(val_loss)) if val_loss else len(train_loss)
    
    # Grid: 1x3 if Seg Acc exists, else 1x2
    has_seg = len(train_seg_acc) > 0
    fig, axes = plt.subplots(1, 3 if has_seg else 2, figsize=(18 if has_seg else 14, 6))
    
    # 1. Loss
    axes[0].plot(epochs[:min_len], train_loss[:min_len], label='Train', color='blue')
    if val_loss: axes[0].plot(epochs[:min_len], val_loss[:min_len], label='Val', color='red', linestyle='--')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Classifier Accuracy
    axes[1].plot(epochs[:min_len], train_clf_acc[:min_len], label='Train', color='blue')
    if val_clf_acc: axes[1].plot(epochs[:min_len], val_clf_acc[:min_len], label='Val', color='green', linestyle='--')
    axes[1].set_title('Classification Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Acc (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Segmentation Accuracy (Optional)
    if has_seg:
        axes[2].plot(epochs[:min_len], train_seg_acc[:min_len], label='Train', color='blue')
        if val_seg_acc: axes[2].plot(epochs[:min_len], val_seg_acc[:min_len], label='Val', color='purple', linestyle='--')
        axes[2].set_title('Segmentation Accuracy')
        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('Acc (%)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[+] Curves saved to {out_path}")
    print(f"Final Train Loss: {train_loss[-1]:.4f} | Clf: {train_clf_acc[-1]:.1f}%" + (f" | Seg: {train_seg_acc[-1]:.1f}%" if has_seg else ""))
    if val_loss:
        print(f"Final Val Loss:   {val_loss[-1]:.4f} | Clf: {val_clf_acc[-1]:.1f}%" + (f" | Seg: {val_seg_acc[-1]:.1f}%" if has_seg else ""))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', nargs='?', default='training_log_dump.txt', help='Path to log file')
    parser.add_argument('--output', default='validation/training_curves_v3.png', help='Output png path')
    args = parser.parse_args()
    parse_and_plot(args.log_file, args.output)
