#!/bin/bash
#SBATCH --job-name=ImgProcess
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=192G
#SBATCH --time=6:00:00
#SBATCH --output=Report_%A-%a.out
#SBATCH --mail-user=hilberthuang05@gatech.edu

cd /storage/coda1/p-fnajafi3/0/yhuang887/Projects/2p_processing_pipeline_202401
source activate suite2p
python run_suite2p_pipeline.py \
--denoise 0 \
--spatial_scale 1 \
--data_path '' \
--save_path '' \
--nchannels 2 \
--functional_chan 2 \
--target_structure 'neuron' \