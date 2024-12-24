# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

import os
import h5py
import numpy as np
import shutil
import argparse
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

def main(args):
    total_frames_per_video = 4089
    video_file_index = (args.target_frame - 1) // total_frames_per_video + 1

    # clear file path.
    if os.path.exists(os.path.join('./', 'results')):
        shutil.rmtree(os.path.join('./', 'results'))
    os.makedirs(os.path.join('./', 'results', 'temp_data'))
    os.makedirs(os.path.join('./', 'results', 'temp_data', 'tiff'))
    os.makedirs(os.path.join('./', 'results', 'temp_model'))
    os.makedirs(os.path.join('./', 'results', 'temp_denoised'))

    # get file names.
    if args.n_channels == 1:
        ch_files = [
            f for f in os.listdir(args.img_path)
            if '_Ch2_'+str(video_file_index).zfill(6)+'.ome.tif' in f]
    if args.n_channels == 2:
        ch_files = [
            f for f in os.listdir(args.img_path)
            if '_Ch1_'+str(video_file_index).zfill(6)+'.ome.tif' in f]
        ch_files += [
            f for f in os.listdir(args.img_path)
            if '_Ch2_'+str(video_file_index).zfill(6)+'.ome.tif' in f]
    # copy to temp folder.
    for file_name in ch_files:
        shutil.copy(
            os.path.join(args.img_path, file_name),
            os.path.join(os.path.join('./', 'results', 'temp_data', 'tiff'), file_name))

    # get dff sub array.
    sig_baseline = 600
    F = np.load(os.path.join(args.suite2p_path, 'suite2p', 'plane0', 'F.npy'), allow_pickle=True)
    Fneu = np.load(os.path.join(args.suite2p_path, 'suite2p', 'plane0', 'Fneu.npy'), allow_pickle=True)
    dff = F.copy() - 0.7*Fneu
    # filtering.
    window_length=9
    polyorder=5
    dff = np.apply_along_axis(
        savgol_filter, 1, dff.copy(),
        window_length=window_length, polyorder=polyorder)
    # baseline correction
    f0 = gaussian_filter(dff, [0., sig_baseline])
    for j in range(dff.shape[0]):
        dff[j,:] = ( dff[j,:] - f0[j,:] ) / f0[j,:]
        dff[j,:] = (dff[j,:] - np.nanmean(dff[j,:])) / (np.nanstd(dff[j,:]) + 1e-5)
    # save dff sub array.
    idx_start = (video_file_index-1)*total_frames_per_video
    idx_end = (video_file_index)*total_frames_per_video
    target_rois = [int(x) for x in args.target_rois.strip('[]').split(',')]
    dff_sub = dff[np.array(target_rois),idx_start:idx_end]
    np.save(os.path.join('./', 'results', 'temp_data', 'dff.npy'), dff_sub)

    # get dff timestamps from voltage recordings.
    f = h5py.File(os.path.join(args.suite2p_path, 'raw_voltages.h5') ,'r')
    vol_time = np.array(f['raw']['vol_time'])
    vol_img = np.array(f['raw']['vol_img'])
    f.close()
    diff_vol = np.diff(vol_img, prepend=0)
    time_img = vol_time[np.where(diff_vol == 1)[0]]
    np.save(os.path.join('./', 'results', 'temp_data', 'time_img.npy'), time_img)
    
    # save roi labels.
    roi_labels = [int(x) for x in args.roi_labels.strip('[]').split(',')]
    np.save(os.path.join('./', 'results', 'temp_data', 'roi_labels.npy'), roi_labels)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='If Farzaneh wants to kill you ask Yicong for a hug!')
    parser.add_argument('--target_frame', required=True, type=int, help='The frame around the output period.')
    parser.add_argument('--target_rois',  required=True, type=str, help='A list of 10 target ROI indice.')
    parser.add_argument('--roi_labels',   required=True, type=str, help='A list of 10 target ROI labels.')
    parser.add_argument('--img_path',     required=True, type=str, help='Path to the session imaging video data.')
    parser.add_argument('--suite2p_path', required=True, type=str, help='Path to the folder of suite2p results.')
    parser.add_argument('--n_channels',   required=True, type=int, help='Number of channels in the session.')
    args = parser.parse_args()
    
    main(args)