#!/bin/bash
#SBATCH --job-name=ppg_train_v2
#SBATCH --output=logs/train_v2_%j.out
#SBATCH --error=logs/train_v2_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenzheng.wang@lip6.fr

# PPG训练 - 防过拟合优化版

echo "=========================================="
echo "PPG分类训练 - 防过拟合优化版"
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
echo "开始训练 - CNN1D模型 (防过拟合配置)"
echo "=========================================="

python3 train_no_overfit.py \
    --data_dir "$DATASET_PATH" \
    --task waveform \
    --model cnn \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.001 \
    --optimizer adamw \
    --scheduler cosine \
    --augment \
    --save_dir checkpoints_cnn_v2

echo -e "\n=========================================="
echo "训练完成！"
echo "结束时间: $(date)"
echo "=========================================="

if [ -f checkpoints_cnn_v2/best_model.pth ]; then
    echo -e "\n最佳模型已保存:"
    ls -lh checkpoints_cnn_v2/best_model.pth
fi
