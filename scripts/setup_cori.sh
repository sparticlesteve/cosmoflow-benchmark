# Source this script to setup the runtime environment on cori
export OMP_NUM_THREADS=32
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"
module load tensorflow/intel-1.13.1-py36
