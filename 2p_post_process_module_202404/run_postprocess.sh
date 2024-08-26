#!/bin/bash
#SBATCH --job-name=ImgProcess
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1:00:00
#SBATCH --output=Report_%A-%a.out
#SBATCH --mail-user=anahri3@gatech.edu

cd /storage/coda1/p-fnajafi3/0/yhuang887/Projects/2p_processing_pipeline_202401
source activate suite2p
python run_postprocess.py \
--session_name 'FN15_P_omi_032124_w' \
--range_skew '0,5' \
--max_connect '3' \
--range_footprint '1,3' \