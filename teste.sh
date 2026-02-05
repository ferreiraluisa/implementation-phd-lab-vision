#!/bin/bash
#SBATCH --partition=A40devel    # GPU partition
#SBATCH --time=1:00:00           # max runtime (HH:MM:SS)
#SBATCH --gpus=1                 # number of GPUs
#SBATCH --ntasks=1               # number of CPU tasks
#SBATCH --output=logs/test.out
#SBATCH --error=logs/test.err

source ~/.bashrc
conda activate h36m

which python
python -V

python -u src/teste.py 
