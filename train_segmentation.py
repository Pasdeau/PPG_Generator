#!/usr/bin/env python3
"""
V2.0 Training Script: Multi-Task UNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import os
from tqdm import tqdm
from ml_training.model_factory import create_model

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    import glob
    if os.path.isfile(args.data_path):
        print("Loading data from single file...")
        data = np.load(args.data_path)
        signals = torch.from_numpy(data['signals']).float()
        masks = torch.from_numpy(data['masks']).long()
        labels = torch.from_numpy(data['labels']).long()
    else:
        print("Loading data from chunks...")
        files = sorted(glob.glob(os.path.join(args.data_path, 'train_data_chunk_*.npz')))
        if not files:
            raise ValueError(f"No train_data_chunk_*.npz files found in {args.data_path}")
        
        all_signals = []
        all_masks = []
        all_labels = []
        for f in tqdm(files, desc="Loading chunks"):
            d = np.load(f)
            all_signals.append(d['signals'])
            all_masks.append(d['masks'])
            all_labels.append(d['labels'])
            
        signals = torch.from_numpy(np.concatenate(all_signals)).float()
        masks = torch.from_numpy(np.concatenate(all_masks)).long()
        labels = torch.from_numpy(np.concatenate(all_labels)).long()

    
    dataset = TensorDataset(signals, masks, labels)
    
    # Split
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 2. Model
    print("Creating UNet...")
    # n_classes_seg = 5 (Clean + 4 Artifacts)
    # n_classes_clf = 5 (Pulse Types)
    model = create_model('unet', input_length=8000, n_classes_seg=5, n_classes_clf=5).to(device)
    
    # 3. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Losses
    criterion_clf = nn.CrossEntropyLoss()
    # Weighted CE for masks (Artifacts are rare compared to background 0)
    # Give higher weight to artifact classes (1-4)
    seg_weights = torch.tensor([1.0, 5.0, 5.0, 5.0, 5.0]).to(device)
    criterion_seg = nn.CrossEntropyLoss(weight=seg_weights)
    
    writer = SummaryWriter(log_dir=args.log_dir)
    best_val_loss = float('inf')
    
    print("Starting Training...")
    for epoch in range(args.epochs):
        # TRAIN
        model.train()
        train_loss = 0
        train_acc_clf = 0
        train_acc_seg = 0
        
        for batch_sig, batch_mask, batch_lbl in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_sig, batch_mask, batch_lbl = batch_sig.to(device), batch_mask.to(device), batch_lbl.to(device)
            
            optimizer.zero_grad()
            pred_clf, pred_seg = model(batch_sig)
            
            # Loss Calculation
            loss_clf = criterion_clf(pred_clf, batch_lbl)
            loss_seg = criterion_seg(pred_seg, batch_mask)
            
            # Mixed Loss (Lambda weights)
            loss = loss_clf + (loss_seg * 10.0) # Segmentation is harder per-pixel
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Metrics
            train_acc_clf += (pred_clf.argmax(1) == batch_lbl).float().mean().item()
            train_acc_seg += (pred_seg.argmax(1) == batch_mask).float().mean().item()
            
        train_loss /= len(train_loader)
        train_acc_clf /= len(train_loader)
        train_acc_seg /= len(train_loader)
        
        # VALIDATE
        model.eval()
        val_loss = 0
        val_acc_clf = 0
        val_acc_seg = 0
        
        with torch.no_grad():
            for batch_sig, batch_mask, batch_lbl in val_loader:
                batch_sig, batch_mask, batch_lbl = batch_sig.to(device), batch_mask.to(device), batch_lbl.to(device)
                
                pred_clf, pred_seg = model(batch_sig)
                
                loss_clf = criterion_clf(pred_clf, batch_lbl)
                loss_seg = criterion_seg(pred_seg, batch_mask)
                loss = loss_clf + (loss_seg * 10.0)
                
                val_loss += loss.item()
                val_acc_clf += (pred_clf.argmax(1) == batch_lbl).float().mean().item()
                val_acc_seg += (pred_seg.argmax(1) == batch_mask).float().mean().item()
        
        val_loss /= len(val_loader)
        val_acc_clf /= len(val_loader)
        val_acc_seg /= len(val_loader)
        
        print(f"  [Train] Loss: {train_loss:.4f} | Classifier Acc: {train_acc_clf:.2%} | Seg Acc: {train_acc_seg:.2%}")
        print(f"  [Val]   Loss: {val_loss:.4f} | Classifier Acc: {val_acc_clf:.2%} | Seg Acc: {val_acc_seg:.2%}")
        
        scheduler.step(val_loss)
        
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/val_clf', val_acc_clf, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_v2.pth'))
            print("  ✓ Saved Best Model")
            
    print("Training Complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to .npz dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='checkpoints_v2')
    parser.add_argument('--log_dir', type=str, default='runs_v2')
    
    args = parser.parse_args()
    train(args)
