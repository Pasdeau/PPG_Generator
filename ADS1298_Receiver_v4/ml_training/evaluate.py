#!/usr/bin/env python3
"""
PPG分类器评估脚本
生成混淆矩阵、分类报告等
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from ml_training.model_factory import create_model
from ml_training.dataset import PPGDataset
from torch.utils.data import DataLoader


def evaluate_model(model, test_loader, device, num_classes):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ 混淆矩阵已保存: {save_path}")


def plot_roc_curves(y_true, y_probs, class_names, save_path):
    """绘制ROC曲线"""
    n_classes = len(class_names)
    
    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # 计算每个类别的ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 绘制
    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ ROC曲线已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='评估PPG分类器')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--data_dir', type=str, default='batch_demo',
                       help='测试数据目录')
    parser.add_argument('--task', type=str, default='waveform',
                       choices=['waveform', 'artifact', 'rhythm'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("PPG分类器评估")
    print("=" * 70)
    
    # 加载配置
    config_path = Path(args.model_path).parent / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"\n加载配置: {config_path}")
        model_type = config['model']
        max_length = config['max_length']
    else:
        print("\n⚠️  未找到配置文件，使用默认值")
        model_type = 'cnn'
        max_length = 8000
    
    # 类别名称
    class_names_dict = {
        'waveform': ['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5'],
        'artifact': ['Clean', 'Device Disp', 'Forearm', 'Hand', 'Poor Contact'],
        'rhythm': ['SR', 'AF']
    }
    class_names = class_names_dict[args.task]
    num_classes = len(class_names)
    
    # 创建模型
    print(f"\n创建模型: {model_type.upper()}")
    model = create_model(model_type, input_length=max_length, num_classes=num_classes)
    
    # 加载权重
    print(f"加载模型权重: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    
    # 加载测试数据
    print(f"\n加载测试数据: {args.data_dir}")
    test_dataset = PPGDataset(args.data_dir, task=args.task, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 评估
    print("\n开始评估...")
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, args.device, num_classes)
    
    # 计算准确率
    accuracy = (y_pred == y_true).mean() * 100
    print(f"\n总体准确率: {accuracy:.2f}%")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 保存分类报告
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_path = output_dir / 'classification_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ 分类报告已保存: {report_path}")
    
    # 绘制混淆矩阵
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    
    # 绘制ROC曲线
    if num_classes > 2:
        roc_path = output_dir / 'roc_curves.png'
        plot_roc_curves(y_true, y_probs, class_names, roc_path)
    
    print("\n" + "=" * 70)
    print("评估完成！")
    print(f"结果保存在: {output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
