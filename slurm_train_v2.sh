#!/bin/bash
#SBATCH --job-name=ppg_train_v2
#SBATCH --output=logs/train_v2_%j.out
#SBATCH --error=logs/train_v2_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenzheng.wang@lip6.fr

# V2.0 训练 (Multi-Task UNet)
echo "=========================================="
echo "PPG V2.0 训练 (UNet Segmentation)"
echo "开始时间: $(date)"
# 模块加载 (根据服务器环境调整)
# module load python/3.8
# module load cuda/11.3
echo "=========================================="

cd ~/ppg_project || exit 1

DATASET_FILE=ml_dataset_v2/train_data.npz

if [ ! -f "$DATASET_FILE" ]; then
    echo "错误: 数据集文件 $DATASET_FILE 不存在"
    exit 1
fi

echo -e "\nGPU列表:"
nvidia-smi

echo -e "\n=========================================="
echo "开始 Multi-Task Training..."
echo "Tasks: Waveform Classification + Artifact Segmentation"
echo "=========================================="

python3 train_segmentation.py \
    --data_path "$DATASET_FILE" \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --save_dir checkpoints_v2 \
    --log_dir runs_v2

echo -e "\n=========================================="
echo "训练完成！"
echo "结束时间: $(date)"
echo "=========================================="
