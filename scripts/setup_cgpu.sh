# Source this script to setup the runtime environment on cori
module load cuda
module load nccl
module load tensorflow/gpu-1.13.1-py36
module load openmpi/3.1.0-ucx

export PATH=$PYTHONUSERBASE/bin:$PATH
