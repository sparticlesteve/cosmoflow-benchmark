#!/bin/bash
#SBATCH -C gpu -c 10
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -J train-cgpu
#SBATCH -o logs/%x-%j.out

# This script defines a training job using pre-staged data on Cori-GPU local NVME.

# Configuration
nTrain=262144
nValid=65536
sourceDir=/global/cscratch1/sd/sfarrell/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf
dataDir=/tmp/sfarrell/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf

# Setup software
. scripts/setup_cgpu.sh
set -e

# Prepare data staging
srun --ntasks-per-node 1 rm -rf $dataDir
srun --ntasks-per-node 1 mkdir -p $dataDir/train $dataDir/validation
date
echo "Pre-staging $nTrain training and $nValid validation samples"
echo "  from $sourceDir"
echo "  into $dataDir"

# Stage data with parallel rsync
#time find $sourceDir -type f | sort | head -n $nStage | \
#    xargs -n1 -P16 -I% rsync -au % $dataDir/

# Stage the data with mpi4py
srun --ntasks-per-node 16 -c 4 -l -u \
    python scripts/stage_data.py -n $nTrain $sourceDir/train $dataDir/train
srun --ntasks-per-node 16 -c 4 -l -u \
    python scripts/stage_data.py -n $nValid $sourceDir/validation $dataDir/validation
date

set -x
#export NCCL_DEBUG=INFO

# Run the training
srun --ntasks-per-node 8 -l -u \
    python train.py -d --rank-gpu \
        --data-dir $dataDir \
        --n-train $nTrain \
        --n-valid $nValid \
        --staged-files 1 $@
