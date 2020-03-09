#!/bin/bash
#COBALT -q default --attrs mcdram=cache:numa=quad
#COBALT -t 30
#COBALT -A datascience
#COBALT -O train-theta
#COBALT -o logs/%x-%j.out


mkdir -p logs
. scripts/setup_theta.sh


NPROC_PER_NODE=4
NPROC=$((NPROC_PER_NODE*COBALT_JOBSIZE))
aprun -n $NPROC -N $NPROC_PER_NODE -e KMP_BLOCKTIME=0 -j 2 -e OMP_NUM_THREADS=32 -cc depth -d 32 python  train.py -d $@
