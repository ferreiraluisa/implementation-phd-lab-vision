#!/bin/bash
#SBATCH --partition=A40devel    # GPU partition
#SBATCH --time=1:00:00           # max runtime (HH:MM:SS)
#SBATCH --gpus=1                 # number of GPUs
#SBATCH --ntasks=1               # number of CPU tasks
#SBATCH --output=logs/train%j.out
#SBATCH --error=logs/train%j.err

source ~/.bashrc
conda activate h36m

which python
python -V

nvidia-smi -l 1800 &

python -u src/train.py \
  --seq-len 40 \
  --root /home/s26ldeso/Human3.6_teste \
  --batch-size 1 \
  --lr 1e-4 \
  --epochs 50 \
  --outdir runs_one_sample/phase1
