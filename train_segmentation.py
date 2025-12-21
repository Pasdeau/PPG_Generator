#!/usr/bin/env python3
"""
V2.0 Training Script: Multi-Task UNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import os
import csv
from tqdm import tqdm
from ml_training.model_factory import create_model
from ml_training.dataset import create_dataloaders

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading data from {args.data_path}...")
    # New generator structure: signals are in data_path/signals
    # If users pass the root 'dataset_v2', append '/signals'
    data_dir = os.path.expanduser(args.data_path)
    if os.path.exists(os.path.join(data_dir, 'signals')):
        data_dir = os.path.join(data_dir, 'signals')
        
    train_loader, val_loader, _ = create_dataloaders(
        data_dir, 
        task='multitask', # Returns (signal, mask, label)
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # 2. Model
    print("Creating UNet...")
    # n_classes_seg = 5 (Clean + 4 Artifacts)
    # n_classes_clf = 5 (Pulse Types)
    # in_channels = 34 (Amp + Vel + 32 CWT)
    model = create_model('unet', input_length=8000, 
                        n_classes_seg=5, n_classes_clf=5, 
                        in_channels=34).to(device)
    
    # 3. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Losses
    criterion_clf = nn.CrossEntropyLoss()
    # Weighted CE for masks (Artifacts are rare compared to background 0)
    # Give higher weight to artifact classes (1-4)
    seg_weights = torch.tensor([1.0, 5.0, 5.0, 5.0, 5.0]).to(device)
    criterion_seg = nn.CrossEntropyLoss(weight=seg_weights)
    
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # CSV Logger
    csv_path = os.path.join(args.save_dir, 'training_log.csv')
    print(f"Logging to {csv_path}")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_acc_clf', 'train_acc_seg', 
                         'val_loss', 'val_acc_clf', 'val_acc_seg', 'lr'])
    
    best_val_loss = float('inf')
    
    print("Starting Training...")
    for epoch in range(args.epochs):
        # TRAIN
        model.train()
        train_loss = 0
        train_acc_clf = 0
        train_acc_seg = 0
        
        # Loader returns (signal, mask, label) because task='multitask'
        for batch_sig, batch_mask, batch_lbl in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_sig = batch_sig.to(device)
            batch_mask = batch_mask.to(device)
            batch_lbl = batch_lbl.to(device)
            
            optimizer.zero_grad()
            pred_clf, pred_seg = model(batch_sig)


            # Loss Calculation
            loss_clf = criterion_clf(pred_clf, batch_lbl)
            loss_seg = criterion_seg(pred_seg, batch_mask)
            
            # Mixed Loss (Weighted)
            loss = (loss_clf * args.lambda_clf) + (loss_seg * args.lambda_seg)
            
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
                batch_sig = batch_sig.to(device)
                batch_mask = batch_mask.to(device)
                batch_lbl = batch_lbl.to(device)
                
                pred_clf, pred_seg = model(batch_sig)
                
                loss_clf = criterion_clf(pred_clf, batch_lbl)
                loss_seg = criterion_seg(pred_seg, batch_mask)
                loss = (loss_clf * args.lambda_clf) + (loss_seg * args.lambda_seg)
                
                val_loss += loss.item()
                val_acc_clf += (pred_clf.argmax(1) == batch_lbl).float().mean().item()
                val_acc_seg += (pred_seg.argmax(1) == batch_mask).float().mean().item()
        
        val_loss /= len(val_loader)
        val_acc_clf /= len(val_loader)
        val_acc_seg /= len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  [Train] Loss: {train_loss:.4f} | Classifier Acc: {train_acc_clf:.2%} | Seg Acc: {train_acc_seg:.2%}")
        print(f"  [Val]   Loss: {val_loss:.4f} | Classifier Acc: {val_acc_clf:.2%} | Seg Acc: {val_acc_seg:.2%}")
        
        scheduler.step(val_loss)
        
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/val_clf', val_acc_clf, epoch)
        
        # Write to CSV
        csv_writer.writerow([epoch+1, train_loss, train_acc_clf, train_acc_seg, 
                             val_loss, val_acc_clf, val_acc_seg, current_lr])
        csv_file.flush()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print("  [INFO] Saved Best Model")
            
    print("Training Complete.")
    csv_file.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='checkpoints_seg')
    parser.add_argument('--log_dir', type=str, default='runs_seg')
    parser.add_argument('--lambda_clf', type=float, default=1.0, help='Weight for classification loss')
    parser.add_argument('--lambda_seg', type=float, default=1.0, help='Weight for segmentation loss')
    
    args = parser.parse_args()
    train(args)
