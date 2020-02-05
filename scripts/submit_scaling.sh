#!/bin/bash

set -ex

# Scaling on Cori KNL up to 1k nodes
sbatch -N 1 scripts/train_cori.sh --data-config "{n_train_files: 1}" configs/scaling.yaml
sbatch -N 2 scripts/train_cori.sh --data-config "{n_train_files: 2}" configs/scaling.yaml
sbatch -N 4 scripts/train_cori.sh --data-config "{n_train_files: 4}" configs/scaling.yaml
sbatch -N 8 scripts/train_cori.sh --data-config "{n_train_files: 8}" configs/scaling.yaml
sbatch -N 16 scripts/train_cori.sh --data-config "{n_train_files: 16}" configs/scaling.yaml
sbatch -N 32 scripts/train_cori.sh --data-config "{n_train_files: 32}" configs/scaling.yaml
sbatch -N 64 -q regular scripts/train_cori.sh --data-config "{n_train_files: 64}" configs/scaling.yaml
sbatch -N 128 -q regular scripts/train_cori.sh --data-config "{n_train_files: 128}" configs/scaling.yaml
sbatch -N 256 -q regular scripts/train_cori.sh --data-config "{n_train_files: 256}" configs/scaling.yaml
sbatch -N 512 -q regular scripts/train_cori.sh --data-config "{n_train_files: 512}" configs/scaling.yaml
sbatch -N 1024 -q regular scripts/train_cori.sh --data-config "{n_train_files: 1024}" configs/scaling.yaml

# Scaling on Cori KNL with dummy data
sbatch -N 1 scripts/train_cori.sh configs/scaling_dummy.yaml
sbatch -N 2 scripts/train_cori.sh configs/scaling_dummy.yaml
sbatch -N 4 scripts/train_cori.sh configs/scaling_dummy.yaml
sbatch -N 8 scripts/train_cori.sh configs/scaling_dummy.yaml
sbatch -N 16 scripts/train_cori.sh configs/scaling_dummy.yaml
sbatch -N 32 scripts/train_cori.sh configs/scaling_dummy.yaml
sbatch -N 64 scripts/train_cori.sh configs/scaling_dummy.yaml
sbatch -N 128 scripts/train_cori.sh configs/scaling_dummy.yaml
sbatch -N 256 scripts/train_cori.sh configs/scaling_dummy.yaml
sbatch -N 512 scripts/train_cori.sh configs/scaling_dummy.yaml
sbatch -N 1024 -q regular scripts/train_cori.sh configs/scaling_dummy.yaml

# Cori GPU scaling
sbatch -J scaling-cgpu --gres=gpu:1 -n 1 scripts/train_cgpu.sh configs/scaling_cgpu.yaml --data-config "{n_train_files: 8}"
sbatch -J scaling-cgpu --gres=gpu:2 -n 2 scripts/train_cgpu.sh configs/scaling_cgpu.yaml --data-config "{n_train_files: 16}"
sbatch -J scaling-cgpu --gres=gpu:4 -n 4 scripts/train_cgpu.sh configs/scaling_cgpu.yaml --data-config "{n_train_files: 32}"
sbatch -J scaling-cgpu --gres=gpu:8 -n 8 scripts/train_cgpu.sh configs/scaling_cgpu.yaml --data-config "{n_train_files: 64}"
sbatch -J scaling-cgpu -N 2 scripts/train_cgpu.sh configs/scaling_cgpu.yaml --data-config "{n_train_files: 128}"
sbatch -q gpu_preempt --requeue -J scaling-cgpu -N 4 scripts/train_cgpu.sh configs/scaling_cgpu.yaml --data-config "{n_train_files: 256}"
sbatch -q gpu_preempt --requeue -J scaling-cgpu -N 8 scripts/train_cgpu.sh configs/scaling_cgpu.yaml --data-config "{n_train_files: 512}"

# Cori GPU scaling with dummy data
sbatch -J scaling-cgpu-dummy --gres=gpu:1 -n 1 scripts/train_cgpu.sh configs/scaling_cgpu_dummy.yaml
sbatch -J scaling-cgpu-dummy --gres=gpu:2 -n 2 scripts/train_cgpu.sh configs/scaling_cgpu_dummy.yaml
sbatch -J scaling-cgpu-dummy --gres=gpu:4 -n 4 scripts/train_cgpu.sh configs/scaling_cgpu_dummy.yaml
sbatch -J scaling-cgpu-dummy --gres=gpu:8 -n 8 scripts/train_cgpu.sh configs/scaling_cgpu_dummy.yaml
sbatch -J scaling-cgpu-dummy -N 2 scripts/train_cgpu.sh configs/scaling_cgpu_dummy.yaml
sbatch -q gpu_preempt --requeue -J scaling-cgpu-dummy -N 4 scripts/train_cgpu.sh configs/scaling_cgpu_dummy.yaml
sbatch -q gpu_preempt --requeue -J scaling-cgpu-dummy -N 8 scripts/train_cgpu.sh configs/scaling_cgpu_dummy.yaml
