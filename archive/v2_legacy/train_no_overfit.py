#!/usr/bin/env python3
"""
PPG分类训练 - 防过拟合优化版
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ml_training.model_factory import create_model
from ml_training.dataset import create_dataloaders
from ml_training.train import Trainer


def main():
    parser = argparse.ArgumentParser(description='PPG分类训练 - 防过拟合版')
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--task', type=str, default='waveform',
                       choices=['waveform', 'artifact', 'rhythm'])
    
    # 模型参数 - 使用更简单的CNN避免过拟合
    parser.add_argument('--model', type=str, default='cnn',
                       help='使用CNN而非ResNet，减少过拟合')
    parser.add_argument('--max_length', type=int, default=8000)
    
    # 训练参数 - 优化以防止过拟合
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32,
                       help='减小batch size增加泛化能力')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='降低学习率 (0.001→0.0001)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                       help='增强L2正则化 (1e-4→1e-3)')
    
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--augment', action='store_true', default=True)
    
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='checkpoints_cnn_waveform')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("PPG分类训练 - 防过拟合优化版")
    print("=" * 70)
    print(f"\n配置 (优化以防止过拟合):")
    print(f"  模型: {args.model.upper()} (更简单，减少过拟合)")
    print(f"  学习率: {args.lr} (降低10倍)")
    print(f"  权重衰减: {args.weight_decay} (增强10倍)")
    print(f"  批次大小: {args.batch_size} (减小)")
    
    num_classes = {'waveform': 5, 'artifact': 5, 'rhythm': 2}[args.task]
    
    print(f"\n加载数据...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        task=args.task,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        augment=args.augment
    )
    
    print(f"\n创建模型: {args.model.upper()}")
    model = create_model(
        args.model,
        input_length=args.max_length,
        num_classes=num_classes
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {num_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"  使用余弦退火学习率调度")
    else:
        scheduler = None
    
    trainer = Trainer(
        model, train_loader, val_loader,
        criterion, optimizer,
        device=args.device,
        save_dir=args.save_dir
    )
    
    if scheduler:
        trainer.scheduler = scheduler
    
    print(f"\n开始训练 (目标: 90%+ 准确率，无过拟合)")
    print(f"预计时间: 1-2小时 (GPU)")
    trainer.train(args.epochs)
    
    print(f"\n最佳模型已保存到: {args.save_dir}/best_model.pth")


if __name__ == '__main__':
    main()
