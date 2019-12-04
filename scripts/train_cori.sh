#!/bin/bash
#SBATCH -C knl
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -J train-cori
#SBATCH -o logs/%x-%j.out

###SBATCH -d singleton

# If using burst buffer
####DW persistentdw name=cosmobb

mkdir -p logs
. scripts/setup_cori.sh

srun -l -u python train.py -d "$@"
