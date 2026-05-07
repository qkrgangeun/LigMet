#!/bin/sh
#SBATCH -J test
#SBATCH -p gpu-super.q
#SBATCH -w nova001
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus 1
#SBATCH -c 2
#SBATCH -o promet_%j.log
#SBATCH -e promet_%j.log
srun python -u /home/qkrgangeun/LigMet/code/scripts/run.py test \
    --config config.yaml --ckpt_path param/dl.ckpt
