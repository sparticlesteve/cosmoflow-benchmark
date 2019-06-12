#!/bin/bash
#SBATCH -C gpu
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH -t 8:00:00
#SBATCH -d singleton
#SBATCH -J train-cgpu
#SBATCH -o logs/%x-%j.out

mkdir -p logs
. scripts/setup_cgpu.sh
srun python train.py $@
