#!/bin/bash
#SBATCH --partition=A40devel    # GPU partition
#SBATCH --time=1:00:00           # max runtime (HH:MM:SS)
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
  --train /home/s26ldeso/Human3.6M_training_data \
  --val /home/s26ldeso/Human3.6M_testing_data \
  --batch-size 16 \
  --lr 1e-4 \
  --epochs 50 \
  --outdir runs_fix_sampler/phase1
