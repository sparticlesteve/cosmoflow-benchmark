#!/bin/bash
#SBATCH -N 256
#SBATCH --ntasks-per-node=2
#SBATCH -n 512
#SBATCH -x "nid[000076-000080,000137-000144]"
####SBATCH --contiguous
#SBATCH -t 6:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --job-name=cosmoGAN_hvd
###SBATCH -C "epyc7742,256gb,epyc7742+256gb"
#SBATCH -C "epyc7742+256gb"
#####SBATCH --nodelist=nid[000256-000287]
#####SBATCH --nodelist=nid[000256-000383]
###SBATCH --nodelist=nid[000256-000512]

###SBATCH --core-spec=2
ulimit -l unlimited
#module load cray-python
source /cray/css/users/jbalma/bin/env_python3.sh
source /cray/css/users/jbalma/bin/setup_env_cuda10.sh
module unload atp
unset PYTHONPATH
module swap cudatoolkit cudatoolkit/10.0.130_3.22-7.0.2.0_5.3__gdfb4ce5
#module load PrgEnv-cray
#module swap PrgEnv-cray PrgEnv-gnu
#module rm gcc
#module load cce
#export MLD_RDK_ENV_INSTALL_DIR=/lus/scratch/jbalma/cuda10.1-conda-py36
#ENV_DIR=/lus/scratch/jbalma/cuda10.0-conda-py36
#conda create -p /lus/scratch/jbalma/conda-py36-cuda100 python=3.6 cudatoolkit=10.0 cudnn
#conda install pip
source activate /lus/scratch/jbalma/conda-py36-cuda100
which python
pip install numpy tensorflow-gpu==1.15
pip install pandas
module load craype-dl-plugin-py3

CC=cc CXX=CC pip install --no-cache-dir --force-reinstall ${CRAYPE_ML_PLUGIN_BASEDIR}/wheel/dl_comm-*.tar.gz
CC=cc CXX=CC pip install --no-cache-dir horovod
#module rm craype-dl-plugin-py3/openmpi/19.12.1.4
module unload craype-dl-plugin-py3
export SCRATCH=/lus/scratch/jbalma


RUN_CMD="python train.py -d configs/scaling_dummy.yaml"
#export MPICH_SMP_SINGLE_COPY_MODE=NONE
#pip install intel-tensorflow==1.15
#module load craype-dl-plugin-py3/19.11.1
#pip uninstall dl_comm
#module load craype-dl-plugin-py3/19.12.1
#pip install $CRAYPE_ML_PLUGIN_BASEDIR/wheel/dl_comm-19.12.1.tar.gz
#module unload craype-dl-plugin-py3/19.12.1 
#module unload craype-dl-plugin-py3/19.11.1

#CC=gcc CXX=g++ pip install /home/crayadm/jbalma/nersc/tmp_inst/craype-dl-plugin-py3/19.12.1.1/wheel/dl_comm-19.12.1.1.tar.gz
#pip uninstall horovod
#cc=cc CXX=CC HOROVOD_HIERARCHICAL_ALLREDUCE=1 HOROVOD_MPICXX_SHOW="CC --cray-print-opts=all" pip install -v --no-cache-dir horovod 
#ulimit -c 

export SCRATCH=/lus/scratch/jbalma
#export CRAY_CUDA_MPS=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CRAY_CUDA_PROXY=1

echo "Running..."
NODES=8 #nodes total
PPN=1 #processer per node
PPS=1 #processes per socket
NP=8 #processes total
NC=18  #job threads per rank
BS=1
#NC=64  #job threads per rank
#IMG_MODE=NCHW
IMG_MODE=NHWC
#NHWC #CPU-only
#NCHW #GPU-only or tensorflow-intel, or tensorflow-cpu + TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

#export OMP_DYNAMIC=TRUE
#export OMP_PROC_BIND=spread
#export OMP_PLACES=cores
#export KMP_AFFINITY=granularity=fine,compact,1,1
#for use with HT
#export KMP_AFFINITY=granularity=course,spread
#for use with cores only
#export KMP_BLOCKTIME=0
#export KMP_TOPOLOGY_METHOD=hwloc
#export KMP_DYNAMIC_MODE=asat
#export OMP_WAIT_POLICY=active
#export KMP_HW_SUBSET=sockets2,numas8,cores128
#export KMP_HW_SUBSET=sockets1,numas4,cores64

export TEMP_DIR=$SCRATCH/temp/cosmoflow-bs${BS}_np${NP}_PPN${PPN}_PPS${PPS}_NC${NC}_Nodes${NODES}_hvd
rm -rf $TEMP_DIR
mkdir -p ${TEMP_DIR}
cp -r ../* ${TEMP_DIR}/
cd ${TEMP_DIR}

export SLURM_WORKING_DIR=${TEMP_DIR}
export PYTHONPATH="$TEMP_DIR:$PYTHONPATH"
#export PYTHONPATH="$(pwd)/data:$(pwd)/utils:$(pwd)/models:$(pwd)/scripts:$PYTHONPATH"
echo $PYTHONPATH

echo "Running ML-Perf HPC WG Cosmoflow Benchmark..."
date

export CUDA_VISIBLE_DEVICES=0
#export TF_FP16_CONV_USE_FP32_COMPUTE=0
#export TF_FP16_MATMUL_USE_FP32_COMPUTE=0
#export HOROVOD_TIMELINE=${SCRATCH_PAD}/timeline.json
#HYPNO="/cray/css/users/kjt/bin/hypno --plot node_power"
#export HOROVOD_FUSION_THRESHOLD=256000000
export HOROVOD_MPI_THREADS_DISABLE=1
#export HOROVOD_FUSION_THRESHOLD=0
#export MPICH_MAX_THREAD_SAFETY=multiple
#export MPICH_COLL_SYNC=1
#export MPICH_ENV_DISPLAY=1

#export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
#export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
#export MKLDNN_VERBOSE=1
#export TF_CPP_MIN_LOG_LEVEL=3
#export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
# clean out the checkpoints
rm -rf ./checkpoints
# these are ignored for GPU runs
export INTER=1
export INTRA=18
#export INTER=1
#export INTRA=0

#export OMP_NUM_THREADS=$INTRA
#export KMP_AFFINITY="granularity=fine,compact,1,0"

#export PYTHONPATH="$(pwd)/data:$(pwd)/utils:$(pwd)/models:$(pwd)/scripts:$PYTHONPATH"
RUN_CMD="python train.py --distributed --rank-gpu configs/scaling_dummy.yaml"

export CRAY_OMP_CHECK_AFFINITY=TRUE
#export KMP_SETTINGS=TRUE

#       DL_COMM_DEFAULT_NTHREADS
#              Default number of threads to create teams with in the case that dl_comm_create_team is not explicitly called.  Defaults to 2.
#       DL_COMM_PIPELINE_CHUNK_KB
#              Size in KB used to transfer data between the host and GPU. Defaults to 256.
#       DL_COMM_NUM_CUDA_STREAMS
#              Integer sets the number of CUDA streams each thread uses for data transfers between the host and GPU.  Using more streams can improve performance.  Defaults to 1.
#       DL_COMM_DEFAULT_PREC_LEVEL
#              Sets precision used for math operations. 0 is floating point. 1 is double. Defaults to 0.

export DL_COMM_DEFAULT_NTHREADS=1

RUN_OPT="-t 1:00:00 -u --cpu_bind=rank_ldom"
time srun -C P100 -n $NP --ntasks-per-node $PPN -N $NODES -u $RUN_OPT $RUN_CMD 2>&1 |& tee ${TEMP_DIR}/logfile

echo "end time = " date
conda deactivate

