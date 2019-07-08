#!/bin/bash
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 5
#BB create_persistent name=cosmobb capacity=5TB access_mode=striped type=scratch 
