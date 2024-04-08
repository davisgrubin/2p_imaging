#!/usr/bin/env python3

import os
import h5py
import numpy as np
from datetime import datetime
from suite2p.registration.register import compute_reference
from suite2p.registration.register import registration_wrapper
from suite2p.io import BinaryFile
from tifffile import imwrite


# make new file to save registered data.

def create_file_to_reg(
        ops,
        ch1_data,
        ch2_data
        ):
    # ch1 registration binary file.
    f_reg_ch1 = BinaryFile(
        Ly=ops['Ly'],
        Lx=ops['Lx'],
        filename=os.path.join(ops['save_path0'], 'temp', 'reg_ch1.bin'),
        n_frames=ch1_data.shape[0])
    # ch2 registration binary file.
    f_reg_ch2 = BinaryFile(
        Ly=ops['Ly'],
        Lx=ops['Lx'],
        filename=os.path.join(ops['save_path0'], 'temp', 'reg_ch2.bin'),
        n_frames=ch2_data.shape[0])
    return f_reg_ch1, f_reg_ch2


# compute max proj image.

def get_proj_img(
        f_reg_ch1,
        f_reg_ch2,
        ops
        ):
    mean_ch1 = np.mean(f_reg_ch1.data, axis=0)
    mean_ch2 = np.mean(f_reg_ch2.data, axis=0)
    frames_ch1 = f_reg_ch1.data[
        np.linspace(0, f_reg_ch1.data.shape[0],
        1 + np.minimum(ops["nimg_init"], f_reg_ch1.data.shape[0]),
        dtype=int)[:-1]]
    frames_ch2 = f_reg_ch2.data[
        np.linspace(0, f_reg_ch2.data.shape[0],
        1 + np.minimum(ops["nimg_init"], f_reg_ch2.data.shape[0]),
        dtype=int)[:-1]]
    ref_ch1 = compute_reference(frames_ch1, ops=ops)
    ref_ch2 = compute_reference(frames_ch2, ops=ops)
    max_ch1 = np.max(f_reg_ch1.data, axis=0)
    max_ch2 = np.max(f_reg_ch2.data, axis=0)
    return [mean_ch1, mean_ch2, ref_ch1, ref_ch2, max_ch1, max_ch2]


# write motion correction results into tiff files.

def save_data_to_tiff(
        f_reg_ch1,
        f_reg_ch2,
        ops
        ):
    imwrite(
        os.path.join(ops['save_path0'], 'raw', 'raw_mc_1.tif'),
        f_reg_ch1.data,
        bigtiff=True)
    imwrite(
        os.path.join(ops['save_path0'], 'raw', 'raw_mc_2.tif'),
        f_reg_ch2.data,
        bigtiff=True)


# save the rigid offsets to h5 file

def save_rigid(
        ops,
        rigid_offsets
        ):
    # file structure:
    # ops['save_path0'] / motion_offsets.h5
    # -- motion_offsets
    # ---- x
    # ---- y
    # ---- z
    f = h5py.File(os.path.join(
        ops['save_path0'], 'raw', 'motion_offsets.h5'), 'w')
    grp = f.create_group('motion_offsets')
    grp['x'] = rigid_offsets[0]
    grp['y'] = rigid_offsets[1]
    grp['z'] = rigid_offsets[2]
    f.close()


# save the mean and max image to h5 file in temp.

def save_proj_img(
        ops,
        mean_ch1, mean_ch2,
        ref_ch1,  ref_ch2,
        max_ch1,  max_ch2
        ):
    # file structure:
    # ops['save_path0'] / temp / proj_img.h5
    # -- proj_img
    # ---- mean_ch1
    # ---- mean_ch2
    # ---- ref_ch1
    # ---- ref_ch2
    # ---- max_ch1
    # ---- max_ch2
    f = h5py.File(os.path.join(
        ops['save_path0'], 'temp', 'proj_img.h5'), 'w')
    grp = f.create_group('proj_img')
    grp['mean_ch1'] = mean_ch1
    grp['mean_ch2'] = mean_ch2
    grp['ref_ch1'] = ref_ch1
    grp['ref_ch2'] = ref_ch2
    grp['max_ch1'] = max_ch1
    grp['max_ch2'] = max_ch2
    f.close()


# main function for motion correction.

def run(
        ops,
        ch1_data, ch2_data
        ):

    print('===============================================')
    print('============== motion correction ==============')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # create registration files.
    [f_reg_ch1, f_reg_ch2] = create_file_to_reg(
        ops, ch1_data, ch2_data)
    print('Registered channel files created in {}'.format(ops['save_path0']))

    if ops['nchannels'] == 1:
        # get functional channel.
        f_reg = f_reg_ch1 if np.sum(ch1_data) > 0 else f_reg_ch2
        f_raw = ch1_data if np.sum(ch1_data) > 0 else ch2_data
        _, _, _, _, rigid_offsets, _, _, _, _, _, _ = registration_wrapper(
            f_reg=f_reg,
            f_raw=f_raw,
            f_reg_chan2=None,
            f_raw_chan2=None,
            ops=ops)
        # dummy channel repeating functional channel.
        if np.sum(ch1_data) == 0:
            f_reg_ch1 = f_reg_ch2
        if np.sum(ch2_data) == 0:
            f_reg_ch2 = f_reg_ch1
    elif ops['nchannels'] == 2:
        _, _, _, _, rigid_offsets, _, _, _, _, _, _ = registration_wrapper(
            f_reg=f_reg_ch1,
            f_raw=ch1_data,
            f_reg_chan2=f_reg_ch2,
            f_raw_chan2=ch2_data,
            ops=ops)
    else:
        raise ValueError('The number of channels is invalid')
    print('Registration completed')
    
    # save offset results.
    save_rigid(ops, rigid_offsets)
    print('Offset results saved.')

    # compute mean and max projection image.
    print('Computing projection images')
    [mean_ch1, mean_ch2, ref_ch1, ref_ch2, max_ch1, max_ch2] = get_proj_img(
        f_reg_ch1, f_reg_ch2, ops)

    # save projection and reference images.
    save_proj_img(
        ops,
        mean_ch1, mean_ch2,
        ref_ch1,  ref_ch2,
        max_ch1,  max_ch2)
    print('Projected images saved')
    
    # write memory mapping numpy to tiff files.
    print('Saving raw motion corrected tiff files')
    save_data_to_tiff(
            f_reg_ch1,
            f_reg_ch2,
            ops
            )
    print('Motion correction tiff results saved')

    return [f_reg_ch1, f_reg_ch2]

