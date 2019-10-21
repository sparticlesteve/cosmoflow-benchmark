# Source this script to setup the runtime environment on cori
export OMP_NUM_THREADS=32
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"
export HDF5_USE_FILE_LOCKING=FALSE
module load tensorflow/intel-1.13.1-py36
module list
