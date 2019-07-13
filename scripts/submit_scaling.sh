#!/bin/bash

set -ex

# Weak scaling dataset size
sbatch -N 1 scripts/train_cori.sh --data-config "{n_train_files: 4}"
sbatch -N 2 scripts/train_cori.sh --data-config "{n_train_files: 8}"
sbatch -N 4 scripts/train_cori.sh --data-config "{n_train_files: 16}"
sbatch -N 8 scripts/train_cori.sh --data-config "{n_train_files: 32}"
sbatch -N 16 scripts/train_cori.sh --data-config "{n_train_files: 64}"
sbatch -N 32 scripts/train_cori.sh --data-config "{n_train_files: 128}"
sbatch -N 64 scripts/train_cori.sh --data-config "{n_train_files: 256}"
