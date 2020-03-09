#!/bin/bash
  
set -ex

# Weak scaling: real dataset
for num_nodes in 1 2 4 8 16 32 64 128 256 512 1024
do

qsub -t 60 -n $num_nodes -A projectname -q default scripts/train_theta.sh --data-config "{n_train_files: $num_nodes}" configs/scaling_theta.yaml

done


# dummy dataset

for num_nodes in 1 2 4 8 16 32 64 128 256 512 1024
do

qsub -t 60 -n $num_nodes -A projectname -q default scripts/train_theta.sh configs/scaling_theta_dummy.yaml

done

