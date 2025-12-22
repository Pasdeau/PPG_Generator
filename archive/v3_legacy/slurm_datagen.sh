#!/bin/bash
#SBATCH --job-name=ppg_datagen
#SBATCH --output=logs/datagen_%j.out
#SBATCH --error=logs/datagen_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wenzheng.wang@lip6.fr

# PPG Training Data Generation Job - Fixed
# Removed incompatible module load commands

echo "=========================================="
echo "PPG Training Data Generation"
echo "Start Time: $(date)"
echo "=========================================="

# Show Python version
echo "Python Version:"
python3 --version

# Running in current directory (where sbatch is submitted)
echo "Running in: $(pwd)"

# Create output directory
mkdir -p ~/ppg_training_data_v3

# Run the Python script
# Using the new Stratified Sampling generator (v2.4+)
# Updated to v3.0 path
python3 generate_training_data.py \
    --num_samples 20000 \
    --output_dir ~/ppg_training_data_v3 \
    --Fd 1000

# Save dataset path for the training job
echo "~/ppg_training_data_v3" > dataset_path.txt

echo -e "\n=========================================="
echo "Data Generation Complete!"
echo "End Time: $(date)"
echo "Dataset Path: ~/ppg_training_data_v3"
echo "Sample Count: $(ls ~/ppg_training_data_v3/signals/*.npz 2>/dev/null | wc -l)"
echo "=========================================="

# Show dataset stats
echo -e "\nDataset Statistics:"
du -sh $DATASET_PATH
ls -lh $DATASET_PATH | head -20
