#!/bin/bash
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 5
#BB create_persistent name=cosmobb capacity=10TB access_mode=striped type=scratch

# Burst buffer docs:
# https://docs.nersc.gov/jobs/examples/#burst-buffer
echo "Creating persistent BB reservation cosmobb"
