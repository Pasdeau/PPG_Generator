#!/usr/bin/env python3
"""
训练集验证脚本
检查数据集质量、完整性和充足性
"""

import numpy as np
import os
import json
from pathlib import Path
from collections import Counter
import sys

def validate_dataset(dataset_dir):
    """验证训练数据集"""
    
    print("=" * 70)
    print("PPG训练集验证")
    print("=" * 70)
    
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ 错误: 数据集目录不存在: {dataset_dir}")
        return False
    
    # 1. 统计文件数量
    print(f"\n[1/6] 文件统计")
    print("-" * 70)
    
    npz_files = list(dataset_path.glob('*.npz'))
    json_files = list(dataset_path.glob('*_meta.json'))
    
    print(f"  .npz 文件数量: {len(npz_files)}")
    print(f"  .json 文件数量: {len(json_files)}")
    
    if len(npz_files) == 0:
        print("  ❌ 没有找到数据文件！")
        return False
    
    # 2. 检查文件完整性
    print(f"\n[2/6] 文件完整性检查")
    print("-" * 70)
    
    valid_samples = 0
    corrupted_files = []
    
    for npz_file in npz_files[:100]:  # 检查前100个
        try:
            data = np.load(npz_file)
            if 'PPG' in data:
                valid_samples += 1
            else:
                corrupted_files.append(npz_file.name)
        except Exception as e:
            corrupted_files.append(npz_file.name)
    
    print(f"  检查样本数: {min(100, len(npz_files))}")
    print(f"  有效样本: {valid_samples}")
    print(f"  损坏文件: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"  ⚠️  发现损坏文件: {corrupted_files[:5]}")
    
    # 3. 分析类别分布
    print(f"\n[3/6] 类别分布分析")
    print("-" * 70)
    
    pulse_types = []
    rhythms = []
    artifact_types = []
    
    for json_file in json_files:
        try:
            with open(json_file) as f:
                meta = json.load(f)
                pulse_types.append(meta.get('pulse_type', 0))
                rhythms.append(meta.get('rhythm', 'unknown'))
                artifact_types.append(meta.get('artifact_type', 'unknown'))
        except:
            continue
    
    pulse_dist = Counter(pulse_types)
    rhythm_dist = Counter(rhythms)
    artifact_dist = Counter(artifact_types)
    
    print(f"  脉搏类型分布:")
    for ptype, count in sorted(pulse_dist.items()):
        percentage = count / len(pulse_types) * 100
        print(f"    Type {ptype}: {count:5d} ({percentage:5.1f}%)")
    
    print(f"\n  心律分布:")
    for rhythm, count in rhythm_dist.items():
        percentage = count / len(rhythms) * 100
        print(f"    {rhythm}: {count:5d} ({percentage:5.1f}%)")
    
    print(f"\n  伪影类型分布:")
    for artifact, count in sorted(artifact_dist.items()):
        percentage = count / len(artifact_types) * 100
        print(f"    {artifact}: {count:5d} ({percentage:5.1f}%)")
    
    # 4. 检查信号质量
    print(f"\n[4/6] 信号质量检查")
    print("-" * 70)
    
    signal_lengths = []
    signal_means = []
    signal_stds = []
    
    for npz_file in npz_files[:100]:
        try:
            data = np.load(npz_file)
            ppg = data['PPG']
            signal_lengths.append(len(ppg))
            signal_means.append(np.mean(ppg))
            signal_stds.append(np.std(ppg))
        except:
            continue
    
    print(f"  信号长度:")
    print(f"    最小: {min(signal_lengths)} 样本")
    print(f"    最大: {max(signal_lengths)} 样本")
    print(f"    平均: {np.mean(signal_lengths):.0f} 样本")
    
    print(f"\n  信号统计:")
    print(f"    平均值范围: [{min(signal_means):.3f}, {max(signal_means):.3f}]")
    print(f"    标准差范围: [{min(signal_stds):.3f}, {max(signal_stds):.3f}]")
    
    # 5. 数据集充足性评估
    print(f"\n[5/6] 数据集充足性评估")
    print("-" * 70)
    
    total_samples = len(npz_files)
    num_classes = len(pulse_dist)
    samples_per_class = total_samples / num_classes if num_classes > 0 else 0
    
    print(f"  总样本数: {total_samples}")
    print(f"  类别数: {num_classes}")
    print(f"  平均每类样本数: {samples_per_class:.0f}")
    
    # 评估标准
    if total_samples < 1000:
        status = "❌ 不足"
        recommendation = "建议至少5,000样本"
    elif total_samples < 5000:
        status = "⚠️  较少"
        recommendation = "建议增加到10,000样本以获得更好效果"
    elif total_samples < 10000:
        status = "✓ 可以"
        recommendation = "数据量足够，可以开始训练"
    else:
        status = "✓✓ 充足"
        recommendation = "数据量充足，预期可获得高准确率"
    
    print(f"\n  评估结果: {status}")
    print(f"  建议: {recommendation}")
    
    # 6. 类别平衡性检查
    print(f"\n[6/6] 类别平衡性检查")
    print("-" * 70)
    
    if pulse_dist:
        max_count = max(pulse_dist.values())
        min_count = min(pulse_dist.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"  最多类别样本数: {max_count}")
        print(f"  最少类别样本数: {min_count}")
        print(f"  不平衡比率: {imbalance_ratio:.2f}")
        
        if imbalance_ratio < 1.5:
            print(f"  ✓ 类别平衡良好")
        elif imbalance_ratio < 3.0:
            print(f"  ⚠️  存在轻微不平衡")
        else:
            print(f"  ❌ 类别严重不平衡，建议重新生成")
    
    # 总结
    print(f"\n{'=' * 70}")
    print("验证总结")
    print("=" * 70)
    
    issues = []
    if len(npz_files) < 5000:
        issues.append("样本数量较少")
    if corrupted_files:
        issues.append(f"{len(corrupted_files)}个文件损坏")
    if pulse_dist and max(pulse_dist.values()) / min(pulse_dist.values()) > 3:
        issues.append("类别严重不平衡")
    
    if not issues:
        print("✓ 数据集验证通过，可以开始训练！")
        print(f"\n推荐训练配置:")
        print(f"  模型: ResNet1D")
        print(f"  Epochs: 100-150")
        print(f"  Batch size: 64")
        print(f"  预期准确率: {85 + min(10, total_samples/2000):.0f}-{90 + min(8, total_samples/2000):.0f}%")
        return True
    else:
        print("⚠️  发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        print(f"\n建议:")
        if len(npz_files) < 5000:
            print(f"  1. 生成更多样本 (当前{len(npz_files)}，建议10,000+)")
        if corrupted_files:
            print(f"  2. 删除损坏的文件")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='验证PPG训练数据集')
    parser.add_argument('--data_dir', type=str, 
                       default='~/ppg_training_data',
                       help='数据集目录')
    
    args = parser.parse_args()
    
    # 展开路径
    data_dir = os.path.expanduser(args.data_dir)
    
    # 验证
    success = validate_dataset(data_dir)
    
    # 返回状态码
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
