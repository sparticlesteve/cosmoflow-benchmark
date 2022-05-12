#!/bin/bash
#SBATCH -J cosmo-pm-shifter
#SBATCH -C gpu
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-task 1
#SBATCH --gpu-bind=none
#SBATCH --image=sfarrell/cosmoflow-gpu:latest
#SBATCH --time 60
#SBATCH -o logs/%x-%j.out

args=$@
export TF_CPP_MIN_LOG_LEVEL=1
env | grep SLURM_NODELIST
set -x

# Run the container
srun -l -u --mpi=pmi2 shifter --module=gpu \
    bash -c "python train.py -d --rank-gpu $args"
