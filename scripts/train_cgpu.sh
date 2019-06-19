#!/bin/bash
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -t 8:00:00
#SBATCH -J train-cgpu
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out

mkdir -p logs
. scripts/setup_cgpu.sh

srun --ntasks-per-node 8 -l -c 10 --mpi=pmi2 \
    python train.py -d --rank-gpu $@
