#!/bin/bash
#SBATCH --job-name=ppg_gen_v2
#SBATCH --output=logs/datagen_v2_%j.out
#SBATCH --error=logs/datagen_v2_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=convergence
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# V2.0 数据生成 (Segmentation Masks + Robustness)
echo "=========================================="
echo "PPG V2.0 数据生成"
echo "开始时间: $(date)"
echo "=========================================="

cd ~/ppg_project || exit 1

# 生成 50,000 样本
python3 generate_segmentation_data.py \
    --num_samples 50000 \
    --output_dir ml_dataset_v2

echo "=========================================="
echo "生成完成！"
echo "结束时间: $(date)"
echo "=========================================="
