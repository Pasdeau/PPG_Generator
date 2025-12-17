#!/bin/bash
#SBATCH --job-name=ppg_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenzheng.wang@lip6.fr

# PPG高精度分类训练作业
# GPU: A100 3g.40gb

echo "=========================================="
echo "PPG分类模型训练"
echo "开始时间: $(date)"
echo "=========================================="

# 进入项目目录
cd ~/ppg_project || exit 1

# 读取数据集路径
if [ -f dataset_path.txt ]; then
    DATASET_PATH=$(cat dataset_path.txt)
    echo "数据集路径: $DATASET_PATH"
else
    DATASET_PATH=~/ppg_training_data
    echo "使用默认数据集路径: $DATASET_PATH"
fi

# 检查数据集
if [ ! -d "$DATASET_PATH" ]; then
    echo "错误: 数据集不存在: $DATASET_PATH"
    exit 1
fi

SAMPLE_COUNT=$(ls $DATASET_PATH/*.npz 2>/dev/null | wc -l)
echo "数据集样本数: $SAMPLE_COUNT"

# 显示GPU信息
echo -e "\nGPU信息:"
nvidia-smi

# 开始训练
echo -e "\n=========================================="
echo "开始训练 - ResNet1D模型"
echo "=========================================="

python3 train_high_accuracy.py \
    --data_dir "$DATASET_PATH" \
    --task waveform \
    --model resnet \
    --epochs 150 \
    --batch_size 64 \
    --lr 0.001 \
    --optimizer adamw \
    --scheduler cosine \
    --augment \
    --save_dir checkpoints_resnet_waveform

# 训练完成
echo -e "\n=========================================="
echo "训练完成！"
echo "结束时间: $(date)"
echo "=========================================="

# 显示最佳模型信息
if [ -f checkpoints_resnet_waveform/best_model.pth ]; then
    echo -e "\n最佳模型已保存:"
    ls -lh checkpoints_resnet_waveform/best_model.pth
    
    if [ -f checkpoints_resnet_waveform/config.json ]; then
        echo -e "\n训练配置:"
        cat checkpoints_resnet_waveform/config.json
    fi
fi
