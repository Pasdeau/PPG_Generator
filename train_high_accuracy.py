#!/usr/bin/env python3
"""
高精度PPG分类训练配置
ResNet1D + 数据增强 + 学习率调度
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ml_training.models import create_model
from ml_training.dataset import create_dataloaders
from ml_training.train import Trainer


def main():
    parser = argparse.ArgumentParser(description='高精度PPG分类训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录')
    parser.add_argument('--task', type=str, default='waveform',
                       choices=['waveform', 'artifact', 'rhythm'])
    
    # 模型参数 - 优化为最高精度
    parser.add_argument('--model', type=str, default='resnet',
                       help='模型类型 (推荐resnet获得最高精度)')
    parser.add_argument('--max_length', type=int, default=8000)
    
    # 训练参数 - 优化配置
    parser.add_argument('--epochs', type=int, default=150,
                       help='训练轮数 (更多轮数获得更好效果)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小 (GPU内存允许的话)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # 优化器和调度器
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['none', 'cosine', 'step'])
    
    # 数据增强
    parser.add_argument('--augment', action='store_true', default=True,
                       help='启用数据增强')
    
    # 其他
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='数据加载线程数')
    parser.add_argument('--save_dir', type=str, default='checkpoints_high_acc')
    
    args = parser.parse_args()
    
    # 打印配置
    print("\n" + "=" * 70)
    print("高精度PPG分类训练")
    print("=" * 70)
    print(f"\n配置 (优化为最高精度):")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # 确定类别数
    num_classes = {
        'waveform': 5,
        'artifact': 5,
        'rhythm': 2
    }[args.task]
    
    # 创建数据加载器
    print(f"\n加载数据...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        task=args.task,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        augment=args.augment
    )
    
    # 创建模型
    print(f"\n创建模型: {args.model.upper()}")
    model = create_model(
        args.model,
        input_length=args.max_length,
        num_classes=num_classes
    )
    
    # 统计参数
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {num_params:,}")
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    
    # 学习率调度器
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"  使用余弦退火学习率调度")
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        print(f"  使用阶梯学习率调度")
    else:
        scheduler = None
    
    # 创建训练器
    trainer = Trainer(
        model, train_loader, val_loader,
        criterion, optimizer,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # 如果有调度器，添加到训练器
    if scheduler:
        trainer.scheduler = scheduler
    
    # 开始训练
    print(f"\n开始训练 (目标: 95%+ 准确率)")
    print(f"预计时间: 2-3小时 (GPU)")
    trainer.train(args.epochs)
    
    print(f"\n最佳模型已保存到: {args.save_dir}/best_model.pth")
    print(f"查看训练日志: tensorboard --logdir=runs")


if __name__ == '__main__':
    main()
