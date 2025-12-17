#!/usr/bin/env python3
"""
PPG分类器训练脚本
支持多种模型和任务
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import sys

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from ml_training.models import create_model
from ml_training.dataset import create_dataloaders


class Trainer:
    """训练器类"""
    
    def __init__(self, model, train_loader, val_loader, 
                 criterion, optimizer, device, save_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir='runs')
        
        # 早停
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for signals, labels in pbar:
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for signals, labels in pbar:
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs):
        """完整训练流程"""
        print("\n" + "=" * 70)
        print("开始训练")
        print("=" * 70)
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            # 打印结果
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=True)
                print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.patience:
                print(f"\n早停触发！验证损失 {self.patience} 个epoch未改善")
                break
            
            # 定期保存
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=False)
        
        print("\n" + "=" * 70)
        print("训练完成！")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print("=" * 70)
        
        self.writer.close()
    
    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description='训练PPG分类器')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='batch_demo',
                       help='数据目录')
    parser.add_argument('--task', type=str, default='waveform',
                       choices=['waveform', 'artifact', 'rhythm'],
                       help='分类任务')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'lstm', 'cnn_lstm', 'resnet'],
                       help='模型类型')
    parser.add_argument('--max_length', type=int, default=8000,
                       help='信号最大长度')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    
    args = parser.parse_args()
    
    # 打印配置
    print("\n" + "=" * 70)
    print("PPG分类器训练")
    print("=" * 70)
    print(f"\n配置:")
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
        num_workers=args.num_workers
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
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 创建训练器
    trainer = Trainer(
        model, train_loader, val_loader,
        criterion, optimizer,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # 开始训练
    trainer.train(args.epochs)
    
    # 保存配置
    config_path = Path(args.save_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n配置已保存到: {config_path}")
    print(f"最佳模型已保存到: {args.save_dir}/best_model.pth")
    print(f"\n查看训练日志: tensorboard --logdir=runs")


if __name__ == '__main__':
    main()
