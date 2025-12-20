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

# PPG High-Precision Training Job
# GPU: A100 3g.40gb

echo "=========================================="
echo "PPG Classification Model Training"
echo "Start Time: $(date)"
echo "=========================================="

# Running in current directory
echo "Running in: $(pwd)"

# Dataset Path
# Try to read from file, otherwise default
if [ -f dataset_path.txt ]; then
    DATASET_PATH=$(cat dataset_path.txt)
else
    DATASET_PATH=~/ppg_training_data_v3
fi
echo "Dataset Path: $DATASET_PATH"

# Check Dataset
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset not found: $DATASET_PATH"
    echo "Please check if data generation completed successfully."
    # Potentially check for 'signals' subdir
    if [ -d "$DATASET_PATH/signals" ]; then
         echo "Found signals directory."
    else
         echo "Warning: signals directory missing?"
    fi
fi

SAMPLE_COUNT=$(ls $DATASET_PATH/signals/*.npz 2>/dev/null | wc -l)
echo "Dataset Sample Count: $SAMPLE_COUNT"

# Show GPU Info
echo -e "\nGPU Info:"
nvidia-smi

# Start Training
echo -e "\n=========================================="
echo "Starting Training - v3.0 CWT Frequency-Enhanced Model"
echo "=========================================="

python3 train_segmentation.py \
    --data_path "$DATASET_PATH" \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --save_dir checkpoints_cwt_v3 \
    --log_dir runs_cwt_v3

# Training Complete
echo -e "\n=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "=========================================="

# Show Best Model Info
if [ -f checkpoints_cwt_v3/best_model.pth ]; then
    echo -e "\nBest Model Saved:"
    ls -lh checkpoints_cwt_v3/best_model.pth
    
    if [ -f checkpoints_cwt_v3/training_log.csv ]; then
        echo -e "\nTraining Log:"
        head -5 checkpoints_cwt_v3/training_log.csv
    fi
fi
