#!/bin/bash
#SBATCH --job-name=Post_processing
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=196G
#SBATCH --time=4:00:00
#SBATCH --output=Report_%A-%a.out
#SBATCH --error=Report_%j.err
#SBATCH --mail-user=alishamsniaaa@gmail.com

# Load Anaconda module (if needed)
module load anaconda3

cd /storage/home/hcoda1/1/ashamsnia6/r-fnajafi3-0/Projects/2AFC_double_block

# Activate the virtual environment
conda activate suite2p_env

python Main.py