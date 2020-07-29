#!/bin/bash
#SBATCH -C knl
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -J train-cori
#SBATCH --image docker:sfarrell/cosmoflow-cpu-mpich:latest
#SBATCH -o logs/%x-%j.out

set -x
srun -l -u shifter python train.py -d $@
