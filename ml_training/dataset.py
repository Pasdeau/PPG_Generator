#!/usr/bin/env python3
"""
PPG数据集加载器
PyTorch Dataset和DataLoader
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from pathlib import Path


class PPGDataset(Dataset):
    """
    PPG信号数据集
    
    支持:
    - 波形分类
    - 伪影分类
    - 心律分类
    """
    
    def __init__(self, data_dir, task='waveform', transform=None, max_length=None):
        """
        Parameters:
        -----------
        data_dir : str
            数据目录路径
        task : str
            'waveform' - 波形分类 (5类)
            'artifact' - 伪影分类 (5类)
            'rhythm' - 心律分类 (2类)
        transform : callable, optional
            数据增强函数
        max_length : int, optional
            信号最大长度（截断或填充）
        """
        self.data_dir = Path(data_dir)
        self.task = task
        self.transform = transform
        self.max_length = max_length
        
        # 加载所有样本
        self.samples = self._load_samples()
        
        # 标签映射
        self.label_maps = {
            'waveform': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},  # pulse_type 1-5 -> 0-4
            'artifact': {
                'clean': 0,
                'device_disp': 1,
                'forearm': 2,
                'hand': 3,
                'poor_contact': 4
            },
            'rhythm': {'SR': 0, 'AF': 1}
        }
    
    def _load_samples(self):
        """加载所有样本文件路径和元数据"""
        samples = []
        
        # 查找所有.npz文件
        npz_files = list(self.data_dir.glob('*.npz'))
        
        for npz_file in npz_files:
            # 查找对应的元数据文件
            meta_file = npz_file.with_name(npz_file.stem + '_meta.json')
            
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                
                samples.append({
                    'npz_path': str(npz_file),
                    'metadata': metadata
                })
            else:
                # 如果没有元数据，从文件名解析
                # 格式: sample_0000_p2_SR_poor_contact.npz
                parts = npz_file.stem.split('_')
                if len(parts) >= 5:
                    samples.append({
                        'npz_path': str(npz_file),
                        'metadata': {
                            'pulse_type': int(parts[2][1]),  # p2 -> 2
                            'rhythm': parts[3],
                            'artifact_type': '_'.join(parts[4:])
                        }
                    })
        
        print(f"加载了 {len(samples)} 个样本")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
        --------
        signal : torch.Tensor
            PPG信号 [1, signal_length]
        label : int
            标签
        """
        sample = self.samples[idx]
        
        # 加载信号
        data = np.load(sample['npz_path'])
        signal = data['PPG']  # 完整PPG信号（含伪影）
        
        # 根据任务选择标签
        metadata = sample['metadata']
        if self.task == 'waveform':
            label = self.label_maps['waveform'][metadata['pulse_type']]
        elif self.task == 'artifact':
            artifact_type = metadata['artifact_type']
            label = self.label_maps['artifact'].get(artifact_type, 0)
        elif self.task == 'rhythm':
            label = self.label_maps['rhythm'][metadata['rhythm']]
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        # 处理信号长度
        if self.max_length is not None:
            if len(signal) > self.max_length:
                # 截断
                signal = signal[:self.max_length]
            elif len(signal) < self.max_length:
                # 填充
                signal = np.pad(signal, (0, self.max_length - len(signal)), mode='constant')
        
        # 转换为tensor
        signal = torch.FloatTensor(signal).unsqueeze(0)  # [1, length]
        label = torch.LongTensor([label])[0]
        
        # 数据增强
        if self.transform:
            signal = self.transform(signal)
        
        return signal, label
    
    def get_class_distribution(self):
        """获取类别分布"""
        from collections import Counter
        
        labels = []
        for sample in self.samples:
            metadata = sample['metadata']
            if self.task == 'waveform':
                labels.append(metadata['pulse_type'])
            elif self.task == 'artifact':
                labels.append(metadata['artifact_type'])
            elif self.task == 'rhythm':
                labels.append(metadata['rhythm'])
        
        return Counter(labels)


class DataAugmentation:
    """数据增强"""
    
    @staticmethod
    def add_noise(signal, noise_level=0.01):
        """添加高斯噪声"""
        noise = torch.randn_like(signal) * noise_level
        return signal + noise
    
    @staticmethod
    def time_shift(signal, shift_range=100):
        """时间平移"""
        shift = np.random.randint(-shift_range, shift_range)
        return torch.roll(signal, shift, dims=-1)
    
    @staticmethod
    def amplitude_scale(signal, scale_range=(0.8, 1.2)):
        """幅度缩放"""
        scale = np.random.uniform(*scale_range)
        return signal * scale
    
    @staticmethod
    def time_stretch(signal, stretch_range=(0.9, 1.1)):
        """时间拉伸（简化版）"""
        # 注意：这是简化实现，实际应用可能需要更复杂的插值
        stretch = np.random.uniform(*stretch_range)
        if stretch == 1.0:
            return signal
        
        # 简单的重采样
        length = signal.shape[-1]
        new_length = int(length * stretch)
        indices = torch.linspace(0, length-1, new_length).long()
        stretched = signal[..., indices]
        
        # 填充或截断到原始长度
        if new_length < length:
            pad = length - new_length
            stretched = torch.nn.functional.pad(stretched, (0, pad))
        else:
            stretched = stretched[..., :length]
        
        return stretched


def create_augmentation_transform(augment_prob=0.5):
    """创建数据增强变换"""
    
    def transform(signal):
        if np.random.rand() < augment_prob:
            # 随机选择增强方法
            aug_type = np.random.choice(['noise', 'shift', 'scale', 'stretch'])
            
            if aug_type == 'noise':
                signal = DataAugmentation.add_noise(signal)
            elif aug_type == 'shift':
                signal = DataAugmentation.time_shift(signal)
            elif aug_type == 'scale':
                signal = DataAugmentation.amplitude_scale(signal)
            elif aug_type == 'stretch':
                signal = DataAugmentation.time_stretch(signal)
        
        return signal
    
    return transform


def create_dataloaders(data_dir, task='waveform', batch_size=32, 
                       train_split=0.7, val_split=0.15, 
                       max_length=8000, num_workers=4, augment=True):
    """
    创建训练、验证、测试数据加载器
    
    Parameters:
    -----------
    data_dir : str
        数据目录
    task : str
        任务类型
    batch_size : int
        批次大小
    train_split : float
        训练集比例
    val_split : float
        验证集比例
    max_length : int
        信号最大长度
    num_workers : int
        数据加载线程数
    augment : bool
        是否使用数据增强
    
    Returns:
    --------
    train_loader, val_loader, test_loader
    """
    
    # 创建完整数据集
    full_dataset = PPGDataset(data_dir, task=task, max_length=max_length)
    
    # 划分数据集
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子
    )
    
    # 为训练集添加数据增强
    if augment:
        # 注意：这里需要修改dataset的transform属性
        # 实际应用中可能需要更复杂的处理
        pass
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {train_size} 样本")
    print(f"  验证集: {val_size} 样本")
    print(f"  测试集: {test_size} 样本")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试数据加载器
    print("=" * 70)
    print("PPG数据集加载器测试")
    print("=" * 70)
    
    # 假设数据在batch_demo目录
    data_dir = '../batch_demo'
    
    if os.path.exists(data_dir):
        # 测试数据集
        dataset = PPGDataset(data_dir, task='waveform', max_length=8000)
        
        print(f"\n数据集大小: {len(dataset)}")
        print(f"\n类别分布: {dataset.get_class_distribution()}")
        
        # 测试单个样本
        signal, label = dataset[0]
        print(f"\n样本形状: {signal.shape}")
        print(f"标签: {label}")
        
        # 测试DataLoader
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir, task='waveform', batch_size=4
        )
        
        # 测试一个batch
        for signals, labels in train_loader:
            print(f"\nBatch形状: {signals.shape}")
            print(f"标签: {labels}")
            break
        
        print("\n✓ 数据加载器测试通过!")
    else:
        print(f"\n⚠️  数据目录不存在: {data_dir}")
        print("请先运行 batch_generate.py 生成数据")
    
    print("=" * 70)
