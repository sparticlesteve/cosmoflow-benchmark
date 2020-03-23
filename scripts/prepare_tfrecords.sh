#!/bin/bash
#SBATCH -J prep-cori
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

maxFiles=8192

set -e
. scripts/setup_cori.sh

# Loop over tasks, 1 per node
for (( i=0; i<$SLURM_JOB_NUM_NODES; i++ )); do
    echo "Launching task $i"
    srun -N 1 -n 1 -c 64 python prepare.py \
        --max-files $maxFiles \
        --n-workers 32 --task $i --n-tasks $SLURM_JOB_NUM_NODES &
done
wait
