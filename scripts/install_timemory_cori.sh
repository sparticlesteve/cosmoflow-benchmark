#!/bin/bash

# This script demonstrates how to do a user-install of timemory
# on the Cori system on top of our TF 1.13.1 installation.

# Setup environment
module load tensorflow/intel-1.13.1-py36
module load gcc/7.3.0
module swap PrgEnv-intel PrgEnv-gnu
module load cmake/3.14.4

# Setup a build directory
TIMEMORY_BUILD_DIR=$SCRATCH/timemory-tf-1.13.1-py36
mkdir -p $TIMEMORY_BUILD_DIR && cd $TIMEMORY_BUILD_DIR

# Do a clean checkout
[ -d timemory ] && rm -rf timemory
git clone https://github.com/NERSC/timemory.git

# Install dependencies missing from the TF installation
pip install --user scikit-build

# Configure the build
export BUILD_SHARED_LIBS=1
export CRAYPE_LINK_TYPE=dynamic

# Run the build
cd timemory
pip install --user .
