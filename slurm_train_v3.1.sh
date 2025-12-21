#!/bin/bash
#SBATCH --job-name=ppg_v3.1_train
#SBATCH --output=logs/train_v3.1_%j.out
#SBATCH --error=logs/train_v3.1_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=convergence

echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"

# Load python environment (Assuming standard conda/venv setup)
# source ~/.bashrc
# conda activate ppg_env
# OR directly use specific python output

DATA_PATH=$(cat dataset_path.txt 2>/dev/null || echo "~/ppg_training_data_v3")
echo "Using Dataset: $DATA_PATH"

# Run Training v3.1 (SE-Attention + Balanced Loss)
python3 train_segmentation.py \
    --data_path "$DATA_PATH" \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.0005 \
    --save_dir "checkpoints_cwt_v3_1" \
    --log_dir "runs_v3_1" \
    --attention \
    --lambda_clf 1.0 \
    --lambda_seg 1.0

echo "Training finished."
