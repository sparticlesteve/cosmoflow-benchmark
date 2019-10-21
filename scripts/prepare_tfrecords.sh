#!/bin/bash
#SBATCH -J prep-cori
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

mkdir -p logs
. scripts/setup_cori.sh

# Loop over tasks, 1 per node
for (( i=0; i<$SLURM_JOB_NUM_NODES; i++ )); do
    echo "Launching task $i"
    srun -N 1 python prepare.py \
        --max-files 4096 \
        --n-workers 32 --task $i --n-tasks $SLURM_JOB_NUM_NODES &
done
wait
