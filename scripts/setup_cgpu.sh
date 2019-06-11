# Source this script to setup the runtime environment on cori
module load cuda
module load tensorflow/gpu-1.13.1-py36
export PATH=$PYTHONUSERBASE/bin:$PATH
