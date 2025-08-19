#!/bin/bash
#SBATCH --job-name=Passive
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=256G
#SBATCH --time=24:00:00
#SBATCH --output=Report_%A-%a.out
#SBATCH --mail-user=hilberthuang05@gatech.edu

cd /storage/coda1/p-fnajafi3/0/yhuang887/Projects/passive_interval_oddball_202412
source activate suite2p
python main.py --config_list 'YH01VT, YH02VT, YH03VT, YH14SC, YH16SC, YH17VT, YH18VT, YH19VT, YH20SC, YH21SC, PPC, V1'
