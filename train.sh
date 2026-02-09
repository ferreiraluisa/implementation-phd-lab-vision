#!/bin/bash
#SBATCH --partition=A40short    # GPU partition
#SBATCH --time=8:00:00           # max runtime (HH:MM:SS)
#SBATCH --gpus=2                 # number of GPUs
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
  --root /home/s26ldeso/Human3.6M_preprocessed_resnet_features \
  --batch-size 16 \
  --lr 1e-4 \
  --epochs 50 \
  --stride 5 \
  --outdir runs/phase1
