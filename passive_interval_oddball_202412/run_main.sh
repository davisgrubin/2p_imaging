#!/bin/bash
#SBATCH --job-name=ImgProcess
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=256G
#SBATCH --time=24:00:00
#SBATCH --output=Report_%A-%a.out
#SBATCH --mail-user=hilberthuang05@gatech.edu

cd /storage/coda1/p-fnajafi3/0/yhuang887/Projects/passive_interval_oddball_202412
source activate suite2p
python main.py
