import matplotlib.pyplot as plt
import re
import os

def parse_and_plot():
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    with open('training_log_dump.txt', 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # Extract metrics using regex
        # [Train] Loss: 7.0059 | Classifier Acc: 99.93% | Seg Acc: 81.66%
        loss_match = re.search(r'Loss:\s*([\d\.]+)', line)
        acc_match = re.search(r'Classifier Acc:\s*([\d\.]+)%', line)
        
        if loss_match and acc_match:
            loss = float(loss_match.group(1))
            acc = float(acc_match.group(1))
            
            if '[Train]' in line:
                train_loss.append(loss)
                train_acc.append(acc)
            elif '[Val]' in line:
                val_loss.append(loss)
                val_acc.append(acc)
                
    # Plotting
    epochs = range(1, len(train_loss) + 1)
    # Adjust valdation length if mistmatch (though typically same)
    min_len = min(len(train_loss), len(val_loss))
    
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs[:min_len], train_loss[:min_len], label='Train Loss', color='blue')
    plt.plot(epochs[:min_len], val_loss[:min_len], label='Val Loss', color='red', linestyle='--')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs (or Steps)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs[:min_len], train_acc[:min_len], label='Train Acc', color='blue')
    plt.plot(epochs[:min_len], val_acc[:min_len], label='Val Acc', color='green', linestyle='--')
    plt.title('Classifier Accuracy')
    plt.xlabel('Epochs (or Steps)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('validation', exist_ok=True)
    out_path = 'validation/training_curves.png'
    plt.savefig(out_path, dpi=150)
    print(f"[+] Curves saved to {out_path}")
    print(f"Final Train Loss: {train_loss[-1]:.4f} | Acc: {train_acc[-1]:.2f}%")
    if val_loss:
        print(f"Final Val Loss:   {val_loss[-1]:.4f} | Acc: {val_acc[-1]:.2f}%")

if __name__ == "__main__":
    parse_and_plot()
