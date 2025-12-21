#!/bin/bash
#SBATCH --job-name=ppg_balanced
#SBATCH --output=logs/train_balanced_%j.out
#SBATCH --error=logs/train_balanced_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenzheng.wang@lip6.fr

# PPG高准确率平衡训练
# ResNet1D + Dropout + 优化超参数

echo "=========================================="
echo "PPG高准确率训练 - 平衡版"
echo "开始时间: $(date)"
echo "=========================================="

cd ~/ppg_project || exit 1

DATASET_PATH=~/ppg_training_data

if [ ! -d "$DATASET_PATH" ]; then
    echo "错误: 数据集不存在"
    exit 1
fi

echo "数据集样本数: $(ls $DATASET_PATH/*.npz 2>/dev/null | wc -l)"
echo -e "\nGPU信息:"
nvidia-smi

echo -e "\n=========================================="
echo "ResNet1D + Dropout(0.5) + 自适应学习率"
echo "目标: 95%+ 准确率，无过拟合"
echo "=========================================="

python3 train_balanced.py \
    --data_dir "$DATASET_PATH" \
    --task waveform \
    --model resnet \
    --epochs 150 \
    --batch_size 64 \
    --lr 0.0005 \
    --weight_decay 0.0005 \
    --optimizer adamw \
    --scheduler plateau \
    --augment \
    --save_dir checkpoints_resnet_balanced

echo -e "\n=========================================="
echo "训练完成！"
echo "结束时间: $(date)"
echo "=========================================="

if [ -f checkpoints_resnet_balanced/best_model.pth ]; then
    echo -e "\n最佳模型已保存:"
    ls -lh checkpoints_resnet_balanced/best_model.pth
    
    if [ -f checkpoints_resnet_balanced/config.json ]; then
        echo -e "\n训练配置:"
        cat checkpoints_resnet_balanced/config.json
    fi
fi
