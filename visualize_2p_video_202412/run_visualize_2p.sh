#!/bin/bash
#SBATCH --job-name=ImgProcess
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -N1 --gres=gpu:H100:4
#SBATCH --mem-per-gpu=8G
#SBATCH --time=6:00:00
#SBATCH --output=Report_%A-%a.out
#SBATCH --mail-user=hilberthuang05@gatech.edu

source activate suite2p
cd /storage/coda1/p-fnajafi3/0/yhuang887/Projects/visualize_2p_video_202412

python get_raw.py \
--target_frame 5000 \
--target_rois '[12,65,2,46,11,51,10,56,26,56]' \
--roi_labels '[1,1,1,0,0,0,0,0,0,0]' \
--img_path './test' \
--suite2p_path './test/FN14_PPC_20241209_seq1131_t' \
--n_channels 2 \

python ./SRDTrans/train.py \
--GPU 0,1 \

python ./SRDTrans/test.py \
--GPU 0,1 \

python get_videos.py \
--labels '[exc, inh]' \
