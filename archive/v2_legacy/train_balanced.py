#!/usr/bin/env python3
"""
PPG分类训练 - 高准确率平衡版
ResNet1D + Dropout + 优化超参数
目标: 95%+ 准确率，无过拟合
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ml_training.model_factory import create_model
from ml_training.dataset import create_dataloaders
from ml_training.train import Trainer


def main():
    parser = argparse.ArgumentParser(description='PPG高准确率训练')
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--task', type=str, default='waveform',
                       choices=['waveform', 'artifact', 'rhythm'])
    
    # 模型参数 - ResNet1D with Dropout
    parser.add_argument('--model', type=str, default='resnet',
                       help='ResNet1D with dropout')
    parser.add_argument('--max_length', type=int, default=8000)
    
    # 优化的训练参数
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='平衡的学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='适中的L2正则化')
    
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       help='使用ReduceLROnPlateau自适应调整')
    parser.add_argument('--augment', action='store_true', default=True)
    
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='checkpoints_resnet_balanced')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("PPG高准确率训练 - 平衡版")
    print("=" * 70)
    print(f"\n配置 (目标: 95%+ 准确率，无过拟合):")
    print(f"  模型: ResNet1D + Dropout(0.5)")
    print(f"  学习率: {args.lr} (平衡)")
    print(f"  权重衰减: {args.weight_decay} (适中)")
    print(f"  学习率调度: ReduceLROnPlateau (自适应)")
    print(f"  数据增强: 启用")
    
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
    
    print(f"\n创建模型: ResNet1D + Dropout")
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
    
    # 使用ReduceLROnPlateau自适应调整学习率
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        print(f"  使用ReduceLROnPlateau (验证损失不降则降低学习率)")
    elif args.scheduler == 'cosine':
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
    
    print(f"\n开始训练")
    print(f"目标: 验证准确率 95%+，训练/验证差距 < 5%")
    print(f"预计时间: 2-3小时 (GPU)")
    trainer.train(args.epochs)
    
    print(f"\n最佳模型已保存到: {args.save_dir}/best_model.pth")


if __name__ == '__main__':
    main()
