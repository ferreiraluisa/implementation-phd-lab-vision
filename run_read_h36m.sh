#!/bin/bash
#SBATCH --partition=A40devel     # GPU partition
#SBATCH --time=1:00:00           # max runtime (HH:MM:SS)
#SBATCH --gpus=1                 # number of GPUs
#SBATCH --ntasks=1               # number of CPU tasks
#SBATCH --cpus-per-task=4        # number of CPU cores per task (optional)
#SBATCH --mem=16G                # RAM (optional)
#SBATCH --output=run_read_h36m_%j.out
#SBATCH --error=run_read_h36m_%j.err

# Load Python module
module load Python

# Activate conda environment if needed
source ~/.bashrc
conda activate h36m

python /home/s26ldeso/implementation-phd-lab-vision/datasets/read_human_36m.py \
    --source_dir /home/s26ldeso/Human3.6 \
    --out_dir /home/s26ldeso/Human3.6M_preprocessed \
    --frame_skip 2
