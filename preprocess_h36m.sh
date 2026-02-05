#!/bin/bash
#SBATCH --partition=A40medium    # GPU partition
#SBATCH --time=24:00:00           # max runtime (HH:MM:SS)
#SBATCH --gpus=4                 # number of GPUs
#SBATCH --ntasks=1               # number of CPU tasks
#SBATCH --output=logs/preprocess.out
#SBATCH --error=logs/preprocess.err

source ~/.bashrc
conda activate h36m

which python
python -V

nvidia-smi -l 1800 &

python -u src/preprocess_resnet_features.py \
  --root /home/s26ldeso/Human3.6M_preprocessed \
  --out /home/s26ldeso/Human3.6M_preprocessed_resnet_features \
  --batch-size 32

