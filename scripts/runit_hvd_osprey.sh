#!/bin/bash
#SBATCH -N 1
##SBATCH --ntasks-per-node=8
##SBATCH -n 8
####SBATCH -x "nid[000076-000080,000137-000144]"
####SBATCH --contiguous
#SBATCH -t 2:00:00
#SBATCH --exclusive
#SBATCH -p spider
#SBATCH -C V100
#SBATCH --job-name=cosmoflow-gpu
#####SBATCH --nodelist=nid[000256-000287]
#####SBATCH --nodelist=nid[000256-000383]
###SBATCH --nodelist=nid[000256-000512]

###SBATCH --core-spec=2
ulimit -l unlimited
source ./config_cuda10.sh
unset PYTHONPATH
#module rm PrgEnv-cray
export SCRATCH=/lus/scratch/jbalma
INSTALL_DIR=/lus/scratch/jbalma/condenv-cuda10-cosmoflow
export CRAY_CPU_TARGET=x86-64
#conda create -y --prefix $INSTALL_DIR python=3.6 cudatoolkit=10.0 cudnn
source activate $INSTALL_DIR
#conda activate $INSTALL_DIR
export PATH=${INSTALL_DIR}/bin:${PATH} #/home/users/${USER}/.local/bin:${PATH}

echo $CUDATOOLKIT_HOME
which gcc
which python
BUILD_TIMEMORY=0

if [ $BUILD_TIMEMORY -eq 0 ]
then
    echo "Running assuming you already have timemory installed"
else

#fi
#pip install tensorflow-gpu==1.15
#Install Horovod on Vanilla Cluster with GPU, OpenMPI support and no NCCL
#CMAKE_CXX_COMPILER=$MPI_CXX CMAKE_CC_COMPILER=$MPI_CC CXX=mpicxx CC=mpicc HOROVOD_CUDA_HOME=${CUDATOOLKIT_HOME} HOROVOD_MPICXX_SHOW="mpicxx -show" HOROVOD_MPI_HOME=${MPI_PATH} HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip install --global-option=build_ext --global-option="-I ${CUDATOOLKIT_HOME}/include" -v --no-cache-dir --force-reinstall horovod

#Install on XC50
#pip uninstall horovod
#cc=cc CXX=CC HOROVOD_HIERARCHICAL_ALLREDUCE=1 HOROVOD_MPICXX_SHOW="CC --cray-print-opts=all" pip install -v --no-cache-dir horovod
#ulimit -c

#Install Horovod on Cluster without GPU support
#CXX=mpic++ CC=gcc HOROVOD_MPICXX_SHOW="mpic++ -show" HOROVOD_MPI_HOME=${MPI_PATH} HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip install --global-option=build_ext --global-option="-I ${CUDATOOLKIT_HOME}/include" --no-cache-dir --force-reinstall horovod

# Setup a build directory
TIMEMORY_BUILD_DIR=$SCRATCH/timemory-tf-1.13.1-py36
rm -rf $TIMEMORY_BUILD_DIR
mkdir -p $TIMEMORY_BUILD_DIR && cd $TIMEMORY_BUILD_DIR

# Do a clean checkout
[ -d timemory ] && rm -rf timemory
git clone https://github.com/NERSC/timemory.git

# Install dependencies missing from the TF installation
python -m pip install scikit-build
python -m pip install pandas
# Configure the build
export BUILD_SHARED_LIBS=1
export CRAYPE_LINK_TYPE=dynamic

# Run the build
cd timemory
python -m pip uninstall timemory
python -m pip install -r requirements.txt --user
#python setup.py install <ARGS> -- <CMAKE_ARGS>
#python setup.py install --help
python -m pip install scikit-build
export CMAKE_CXX_COMPILER=$MPI_CXX
export CMAKE_CC_COMPILER=$MPI_CC
python setup.py install --enable-gotcha --enable-mpi -- -DTIMEMORY_BUILD_TOOLS=OFF -DCMAKE_CXX_COMPILER=$MPI_CXX -DCMAKE_CC_COMPILER=$MPI_CC -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF

fi

cd /cray/css/users/jbalma/Innovation-Proposals/ML-Perf/hpc-wg/Cosmoflow/cosmoflow-benchmark/scripts

RUN_CMD="python train.py -d configs/scaling_dummy.yaml"
#
#export CRAY_CUDA_MPS=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CRAY_CUDA_PROXY=1

echo "Running..."
NODES=1 #nodes total
PPN=8 #processer per node
PPS=1 #processes per socket
NP=8 #processes total
NC=9  #job threads per rank
BS=1
#NC=64  #job threads per rank
#IMG_MODE=NCHW
IMG_MODE=NHWC

export RUN_NAME=mlperf-cosmoflow-bs${BS}_np${NP}_PPN${PPN}_PPS${PPS}_NC${NC}_Nodes${NODES}_hvd
export TEMP_DIR=$SCRATCH/temp/${RUN_NAME}
rm -rf $TEMP_DIR
mkdir -p ${TEMP_DIR}
cp -r ../* ${TEMP_DIR}/
cd ${TEMP_DIR}

export SLURM_WORKING_DIR=${TEMP_DIR}
#export PYTHONPATH="$TEMP_DIR:$PYTHONPATH"
#export PYTHONPATH="$(pwd)/data:$(pwd)/utils:$(pwd)/models:$(pwd)/scripts:$PYTHONPATH"
echo $PYTHONPATH

echo "Running ML-Perf HPC WG Cosmoflow Benchmark..."
date

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export TF_FP16_CONV_USE_FP32_COMPUTE=0
#export TF_FP16_MATMUL_USE_FP32_COMPUTE=0
#export HOROVOD_TIMELINE=${SCRATCH_PAD}/timeline.json
#export HOROVOD_FUSION_THRESHOLD=256000000
#export HOROVOD_MPI_THREADS_DISABLE=1
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
export INTRA=${NC}

#export OMP_NUM_THREADS=$INTRA
#export KMP_AFFINITY="granularity=fine,compact,1,0"

export PYTHONPATH="${TEMP_DIR}:${TEMP_DIR}/data:${TEMP_DIR}/models:${TEMP_DIR}/scripts:$PYTHONPATH"
pwd


export HOROVOD_CACHE_CAPACITY=0
#export HOROVOD_CACHE_CAPACITY=16384
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export OMPI_MCA_btl_openib_allow_ib=false
export OMPI_MCA_btl_openib_allow_ib=true
export OMPI_MCA_btl=^openib
#export UCX_TLS="cma,dc_mlx5,posix,rc,rc_mlx5,self,sm,sysv,tcp,ud,ud_mlx5"
#export UCX_MEMTYPE_CACHE=n
#export UCX_ACC_DEVICES=""
#export UCX_NET_DEVICES="ib0,eth0,mlx5_0:1" #,ib0,eth0"   #mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
#export DL_COMM_USE_CRCCL=1
#export OMPI_MCA_btl_tcp_if_include=ib0
#-mca btl_tcp_if_include ens4d1


RUN_CMD="python train.py --distributed --rank-gpu -v configs/scaling_dummy_cray.yaml"

export CRAY_OMP_CHECK_AFFINITY=TRUE
#
#        "TIMEMORY_MPI_INIT": false,
#        "TIMEMORY_MPI_FINALIZE": false,
#        "TIMEMORY_MPI_THREAD": false,
#        "TIMEMORY_MPI_THREAD_TYPE": "",

export TIMEMORY_UPCXX_INIT=false
export TIMEMORY_UPCXX_FINALIZE=false


#RUN_OPT="-t 1:00:00 -u --cpu_bind=rank_ldom"
#srun -C V100 -p spider -n $NP --ntasks-per-node $PPN -N $NODES -u $RUN_OPT $RUN_CMD 2>&1 |& tee ${TEMP_DIR}/logfile
srun -p spider --accel-bind=g,v --cpu_bind=none -c ${NC} -C V100 -l -N ${NODES} -n ${NP} --ntasks-per-node=${PPN} -u $RUN_CMD 2>&1 |& tee ${TEMP_DIR}/logfile

echo "end time = " date
conda deactivate

