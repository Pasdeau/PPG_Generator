#!/bin/bash
#SBATCH --job-name=ppg_datagen
#SBATCH --output=logs/datagen_%j.out
#SBATCH --error=logs/datagen_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenzheng.wang@lip6.fr

# PPG训练数据生成作业 - 修复版
# 移除了不兼容的module load命令

echo "=========================================="
echo "PPG训练数据生成"
echo "开始时间: $(date)"
echo "=========================================="

# 显示Python版本
echo "Python版本:"
python3 --version

# 进入项目目录
cd ~/ppg_project || exit 1

# 创建输出目录
mkdir -p ~/ppg_training_data

# 生成数据集
echo -e "\n生成20,000样本数据集..."
python3 batch_generate.py \
    --num_samples 20000 \
    --output_dir ~/ppg_training_data

# 保存数据集路径
DATASET_PATH=~/ppg_training_data
echo "$DATASET_PATH" > dataset_path.txt

echo -e "\n=========================================="
echo "数据生成完成！"
echo "结束时间: $(date)"
echo "数据集路径: $DATASET_PATH"
echo "样本数量: $(ls $DATASET_PATH/*.npz 2>/dev/null | wc -l)"
echo "=========================================="

# 显示数据集统计
echo -e "\n数据集统计:"
du -sh $DATASET_PATH
ls -lh $DATASET_PATH | head -20
