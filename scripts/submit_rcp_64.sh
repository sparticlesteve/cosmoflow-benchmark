#!/bin/bash

# This script contains the commands used to submit the RCP-64 runs on Cori-GPU

module purge
module load cgpu

#-------------------------------------------------------------------------------
# bs 64
sbatch --array="1-2%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-01 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-2%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-02 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-03 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-04 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-05 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-06 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-07 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-08 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-09 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-10 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-11 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-12 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-13 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-14 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-15 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-16 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-17 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-18 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-19 \
    --stage-dir /tmp/cosmo --resume

sbatch --array="1-4%1" -N 4 --time-min 4:00:00 --time 24:00:00 \
    scripts/train_cgpu.sh configs/cosmo_rcp_64.yaml --batch-size 2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/cgpu-rcp-64-20 \
    --stage-dir /tmp/cosmo --resume
