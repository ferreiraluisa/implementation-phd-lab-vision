#!/bin/bash
#SBATCH --partition=medium    # GPU partition
#SBATCH --time=24:00:00           # max runtime (HH:MM:SS)
#SBATCH --gpus=1                 # number of GPUs
#SBATCH --ntasks=1               # number of CPU tasks
#SBATCH --output=logs/train%j.out
#SBATCH --error=logs/train%j.err

source ~/.bashrc
conda activate h36m

which python
python -V

python -u src/train.py \
  --seq-len 40 \
  --batch-size 16 \
  --lr 1e-4 \
  --epochs 50 \
  --stride 5 \
  --outdir runs/phase1
