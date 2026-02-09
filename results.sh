#!/bin/bash
#SBATCH --partition=A40short    # GPU partition
#SBATCH --time=8:00:00           # max runtime (HH:MM:SS)
#SBATCH --gpus=1                # number of GPUs
#SBATCH --ntasks=1               # number of CPU tasks
#SBATCH --output=logs/results.out
#SBATCH --error=logs/results.err

source ~/.bashrc
conda activate h36m

which python
python -V

nvidia-smi -l 1800 &

python -u src/results.py \
  --features_root /home/s26ldeso/Human3.6M_preprocessed_resnet_features \
  --preprocessed_root /home/s26ldeso/Human3.6M_preprocessed \
  --model_path /home/s26ldeso/implementation-phd-lab-vision/runs/phase1/best.pt \
  --out outputs/example_result_S9.npz