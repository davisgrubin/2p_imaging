#!/bin/bash
#SBATCH --job-name=ImgProcess
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -N1 --gres=gpu:V100:2
#SBATCH --mem-per-gpu=32G
#SBATCH --time=12:00:00
#SBATCH --output=Report_%A-%a.out
#SBATCH --mail-user=hilberthuang05@gatech.edu

source activate suite2p
cd /storage/coda1/p-fnajafi3/0/yhuang887/Projects/visualize_2p_video_202412

python get_raw.py \
--target_frame 25000 \
--target_rois '[1,2,27,15,12,14,3,98,6,22,11,19]' \
--roi_labels '[1,1,1,0,0,0,0,0,0,0,0,0]' \
--img_path '/storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging/FN14/20240605/FN14_P_20240605_seq1420_t-583' \
--suite2p_path '/storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging/processed/passive/sequence_omission/FN14/FN14_P_20240605_seq1420_t' \
--n_channels 2 \

python ./SRDTrans/train.py \
--datasets_path './results/temp_data/tiff' \
--datasets_folder './' \
--pth_path './results/temp_model' \
--n_epochs 8 \
--GPU 0,1 \
--train_datasets_size 8192 \
--patch_x 160 \
--patch_t 160 \

python ./SRDTrans/test.py \
--datasets_path './results/temp_data/tiff' \
--datasets_folder './' \
--pth_path './results/temp_model' \
--denoise_model './' \
--output_path './results/temp_denoised' \
--GPU 0,1 \
--patch_x 160 \
--patch_t 160 \

python get_videos.py \
--labels '[exc, inh]' \
