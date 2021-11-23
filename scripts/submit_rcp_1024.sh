#!/bin/bash

# This script contains the commands used to submit the RCP-1024 runs on
# Perlmutter

#-------------------------------------------------------------------------------
# BS 1024, 64 nodes, 256 gpus
sbatch -n 256 -t 4:00:00 \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-01 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-02 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-03 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-04 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-05 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-06 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-07 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-08 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-09 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-10 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-11 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-12 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-13 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-14 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-15 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-16 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-17 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-18 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-19 \
    --batch-size 4 --resume --mlperf
sbatch -n 256 -t 4:00:00 -d singleton \
    scripts/train_pm_shifter.sh configs/cosmo_rcp_1024.yaml \
    --data-dir $SCRATCH/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf_v2 \
    --output-dir $SCRATCH/cosmoflow-benchmark/results-rcp/pm-rcp-1024-20 \
    --batch-size 4 --resume --mlperf
