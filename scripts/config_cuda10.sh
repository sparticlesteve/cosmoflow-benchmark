# Source me

export SCRATCH=/lus/scratch/$USER

if [ $USER == "swowner" ]; then
    umask 002 # all-readable
    INSTALL_BASE=/usr/common/software
else
    INSTALL_BASE=$SCRATCH/condaenv
fi
#export INSTALL_BASE=$SCRATCH/conda
# Configure the installation
export INSTALL_DIR=/lus/scratch/jbalma/condenv-cuda10-cosmoflow

# Setup programming environment
#module unload PrgEnv-cray
#module load PrgEnv-gnu
#module unload atp
#module unload cray-libsci
#module unload craype-hugepages8M
#module unload craype-broadwell
#module unload gcc
#module load gcc/7.2.0
source /cray/css/users/jbalma/bin/setup_env_cuda10_osprey_V100.sh
source /cray/css/users/jbalma/bin/env_python3.sh
export CRAY_CPU_TARGET=x86-64

# Setup conda
#source /usr/common/software/python/3.6-anaconda-5.2/etc/profile.d/conda.sh
#source /cray/css/users/dctools/anaconda3/etc/profile.d/conda.sh
#conda create $INSTALL_DIR

# Print some stuff
echo "Configuring on $(hostname) as $USER"
echo "  Install directory $INSTALL_DIR"
