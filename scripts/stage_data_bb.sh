#!/bin/bash
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 5
#DW persistentdw name=cosmobb
#DW stage_in source=/global/cscratch1/sd/sfarrell/cosmoflow-benchmark/data/cosmoUniverse_2019_02_4parE destination=$DW_PERSISTENT_STRIPED_cosmobb/cosmoUniverse_2019_02_4parE type=directory

echo "Yessss"
echo $DW_PERSISTENT_STRIPED_cosmobb
ls $DW_PERSISTENT_STRIPED_cosmobb
