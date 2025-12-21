#!/bin/bash
#SBATCH --job-name=ppg_v4.0_dual
#SBATCH --output=/home/users/wang/logs/train_v4.0_%j.out
#SBATCH --error=/home/users/wang/logs/train_v4.0_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --partition=NGA100
#SBATCH --mem=32G

# Environments
module load anaconda3/2023.03
source activate ppg_env

# Variables (Updated for Remote Paths)
DATA_DIR="$HOME/ppg_training_data_v3"
SAVE_DIR="checkpoints_dual_v4"
LOG_DIR="runs_dual_v4"

# Create directories
mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR
mkdir -p /home/users/wang/logs

echo "Starting v4.0 Dual-Stream Training on $(hostname)"
echo "Date: $(date)"

# Run Training Script
# Uses train_dual_stream.py which defaults to 'dual' model
python3 train_dual_stream.py \
    --data_path "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --log_dir "$LOG_DIR" \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.0005 \
    --lambda_clf 1.0 \
    --lambda_seg 1.0

echo "Training finished at $(date)"
