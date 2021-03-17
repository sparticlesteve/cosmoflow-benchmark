#!/bin/bash
#SBATCH -J prep-cori
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -N 64
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

maxFiles=10000

set -e
. scripts/setup_cori.sh

outputDir=/global/cscratch1/sd/sfarrell/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2

# Loop over tasks, 1 per node
for (( i=0; i<$SLURM_JOB_NUM_NODES; i++ )); do
    echo "Launching task $i"
    srun -N 1 -n 1 -c 64 python prepare.py \
        --max-files $maxFiles --output-dir $outputDir --write-tfrecord --gzip \
        --n-workers 32 --task $i --n-tasks $SLURM_JOB_NUM_NODES &
done
wait
