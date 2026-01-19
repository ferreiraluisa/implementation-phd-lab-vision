#!/bin/bash
#SBATCH --partition=A40devel     # GPU partition
#SBATCH --time=1:00:00           # max runtime (HH:MM:SS)
#SBATCH --gpus=1                 # number of GPUs
#SBATCH --ntasks=1               # number of CPU tasks
#SBATCH --cpus-per-task=4        # number of CPU cores per task (optional)
#SBATCH --mem=16G                # RAM (optional)

# Load Python module
module load Python

# Activate conda environment if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv

python /home/s26ldeso/Human3.6M/src/datasets/h36/read_human36m.py \
    --source_dir /home/s26ldeso/Human3.6 \
    --out_dir /home/s26ldeso/Human3.6M_preprocessed \
    --frame_skip 2
