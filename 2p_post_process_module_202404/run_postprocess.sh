#!/bin/bash
#SBATCH --job-name=ImgProcess
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1:00:00
#SBATCH --output=Report_%A-%a.out
#SBATCH --mail-user=grubin6@gatech.edu

cd /storage/home/hcoda1/3/grubin6/2p_imaging/2p_post_process_module_202404
source activate suite2p
python run_postprocess.py \
--session_data_path="/storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging/processed/passive/YH01VT/VTYH01_PPC_20250113_1451ShortLong" \
--range_skew="-5.0,5.0" \
--range_aspect="0.0,5.0" \
--max_connect=1 \
--range_footprint="1.0,2.0" \
--range_compact="0.0,1.06" \
--diameter=6
