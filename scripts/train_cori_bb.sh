#!/bin/bash
#SBATCH -C knl
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -J train-cori
#SBATCH -o logs/%x-%j.out
#DW persistentdw name=cosmobb

. scripts/setup_cori.sh

set -x
srun -l -u python train.py \
    --data-dir $DW_PERSISTENT_STRIPED_cosmobb/cosmoUniverse_2019_05_4parE_tf \
    -d $@
