#!/bin/bash
#SBATCH --job-name=ImgProcess
#SBATCH --account=gts-fnajafi3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -N1 --gres=gpu:V100:1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=12:00:00
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
--datasets_path '.\results\temp_data\tiff' \
--datasets_folder '.\' \
--pth_path '.\results\temp_model' \
--n_epochs 20 \
--GPU 0 \
--train_datasets_size 6000 \
--patch_x 160 \
--patch_t 160 \

python ./SRDTrans/test.py \
--datasets_path '.\results\temp_data\tiff' \
--datasets_folder '.\' \
--pth_path '.\results\temp_model' \
--denoise_model '.\' \
--output_path '.\results\temp_denoised' \
--GPU 0 \
--patch_x 160 \
--patch_t 160 \

python get_videos.py \
--labels '[exc, inh]' \
