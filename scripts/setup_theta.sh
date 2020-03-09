# Source this script to setup the runtime environment on Theta
export OMP_NUM_THREADS=32
export KMP_BLOCKTIME=0
export KMP_AFFINITY="granularity=fine,compact,1,0"
export HDF5_USE_FILE_LOCKING=FALSE

## loads Intel Python 3.6, Tensorflow 1.14 and Horovod 0.16.4
module load datascience/tensorflow-1.14
