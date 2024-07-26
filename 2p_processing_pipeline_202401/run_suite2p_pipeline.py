#!/usr/bin/env python3
import os
import json
import h5py
import shutil
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from suite2p import run_s2p


'''
python run_suite2p_pipeline.py `
--denoise 0 `
--spatial_scale 1 `
--data_path './testdata/C1' `
--save_path './results/C1' `
--nchannels 1 `
--functional_chan 2 `
--brain_region 'crbl' `
'''


# setting the ops.npy for suite2p.

def set_params(args):

    print('===============================================')
    print('============ configuring parameters ===========')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # read the structured config json file.
    if args.brain_region not in ['ppc', 'crbl']:
        raise ValueError('brain_region can only be ppc or crbl')
    elif args.brain_region == 'ppc':
        with open('./config_ppc.json', 'r') as file:
            params = json.load(file)
    elif args.brain_region == 'crbl':
        with open('./config_crbl.json', 'r') as file:
            params = json.load(file)

    # convert to ops.npy for suite2p by removing the first layer.
    ops = dict()
    for key in params.keys():
        ops.update(params[key])

    # create the path for saving data.
    if not os.path.exists(os.path.join(args.save_path)):
        os.makedirs(os.path.join(args.save_path))
    if not os.path.exists(os.path.join(args.save_path, 'figures')):
        os.makedirs(os.path.join(args.save_path, 'figures'))

    # set params specified by command line.
    ops['denoise']         = args.denoise
    ops['spatial_scale']   = args.spatial_scale
    ops['data_path']       = args.data_path
    ops['save_path0']      = args.save_path
    ops['nchannels']       = args.nchannels
    ops['functional_chan'] = args.functional_chan
    ops['align_by_chan']   = 3-args.functional_chan
    ops['brain_region']    = args.brain_region
    print('Search data files in {}'.format(ops['data_path']))
    print('Will save processed data in {}'.format(ops['save_path0']))
    print('Processing {} channel data'.format(ops['nchannels']))
    print('Set functional channel to ch'+str(ops['functional_chan']))

    # set db for suite2p.
    db = {
        'data_path' : [args.data_path],
        'save_path0' : args.save_path,
        }

    # save ops.npy to the path.
    print('Parameters setup for {} completed'.format(args.brain_region))

    return ops, db


# processing voltage recordings.

def process_vol(args):

    # read the voltage recording file.
    def read_vol_to_np(
            args,
            ):
        # voltage: SESSION_Cycle00001_VoltageRecording_000NUM.csv.
        vol_record = [f for f in os.listdir(ops['data_path'])
                      if 'VoltageRecording' in f and '.csv' in f]
        df_vol = pd.read_csv(
            os.path.join(args.data_path, vol_record[0]),
            engine='python')
        # time index in ms.
        vol_time  = df_vol['Time(ms)'].to_numpy()
        # AI0: Bpod BNC1 (trial start signal from bpod).
        if ' Input 0' in df_vol.columns.tolist():
            vol_start = df_vol[' Input 0'].to_numpy()
        else:
            vol_start = np.zeros_like(vol_time)
        # AI1: sync patch and photodiode (visual stimulus).
        if ' Input 1' in df_vol.columns.tolist():
            vol_stim_vis = df_vol[' Input 1'].to_numpy()
        else:
            vol_stim_vis = np.zeros_like(vol_time)
        # AI2: HIFI BNC output.
        if ' Input 2' in df_vol.columns.tolist():
            vol_hifi = df_vol[' Input 2'].to_numpy()
        else:
            vol_hifi = np.zeros_like(vol_time)
        # AI3: ETL scope imaging output (2p microscope image trigger signal).
        if ' Input 3' in df_vol.columns.tolist():
            vol_img = df_vol[' Input 3'].to_numpy()
        else:
            vol_img = np.zeros_like(vol_time)
        # AI4: Hifi audio output waveform (HIFI waveform signal).
        if ' Input 4' in df_vol.columns.tolist():
            vol_stim_aud = df_vol[' Input 4'].to_numpy()
        else:
            vol_stim_aud = np.zeros_like(vol_time)
        # AI5: FLIR output.
        if ' Input 5' in df_vol.columns.tolist():
            vol_flir = df_vol[' Input 5'].to_numpy()
        else:
            vol_flir = np.zeros_like(vol_time)
        # AI6: PMT shutter.
        if ' Input 6' in df_vol.columns.tolist():
            vol_pmt = df_vol[' Input 6'].to_numpy()
        else:
            vol_pmt = np.zeros_like(vol_time)
        # AI7: PMT shutter.
        if ' Input 7' in df_vol.columns.tolist():
            vol_led = df_vol[' Input 7'].to_numpy()
        else:
            vol_led = np.zeros_like(vol_time)
        vol = {
            'vol_time'     : vol_time,
            'vol_start'    : vol_start,
            'vol_stim_vis' : vol_stim_vis,
            'vol_hifi'     : vol_hifi,
            'vol_img'      : vol_img,
            'vol_stim_aud' : vol_stim_aud,
            'vol_flir'     : vol_flir,
            'vol_pmt'      : vol_pmt,
            'vol_led'      : vol_led
            }
        return vol

    # threshold the continuous voltage recordings to 01 series.
    def thres_binary(
            data,
            thres
            ):
        data_bin = data.copy()
        data_bin[data_bin<thres] = 0
        data_bin[data_bin>thres] = 1
        return data_bin

    # convert all voltage recordings to binary series.
    def vol_to_binary(vol):
        vol['vol_start']    = thres_binary(vol['vol_start'],    1)
        vol['vol_stim_vis'] = thres_binary(vol['vol_stim_vis'], 1)
        vol['vol_hifi']     = thres_binary(vol['vol_hifi'],     0.5)
        vol['vol_img']      = thres_binary(vol['vol_img'],      1)
        vol['vol_pmt']      = thres_binary(vol['vol_pmt'],      1)
        vol['vol_flir']     = thres_binary(vol['vol_flir'],     1)
        vol['vol_led']      = thres_binary(vol['vol_led'],     1)
        return vol

    # save voltage data.
    def save_vol(args, vol):
        # file structure:
        # args.save_path / raw_voltages.h5
        # -- raw
        # ---- vol_time
        # ---- vol_start_bin
        # ---- vol_stim_vis
        # ---- vol_img_bin
        f = h5py.File(os.path.join(
            args.save_path, 'raw_voltages.h5'), 'w')
        grp = f.create_group('raw')
        grp['vol_time']      = vol['vol_time']
        grp['vol_start']     = vol['vol_start']
        grp['vol_stim_vis']  = vol['vol_stim_vis']
        grp['vol_hifi']      = vol['vol_hifi']
        grp['vol_img']       = vol['vol_img']
        grp['vol_stim_aud']  = vol['vol_stim_aud']
        grp['vol_flir']      = vol['vol_flir']
        grp['vol_pmt']       = vol['vol_pmt']
        grp['vol_led']       = vol['vol_led']
        f.close()

    # run processing.
    try:
        vol = read_vol_to_np(args)
        vol = vol_to_binary(vol)
        save_vol(args, vol)
    except:
        print('Valid voltage recordings csv file not found')


# move bpod session data.

def move_bpod_mat(args):
    bpod_mat = [f for f in os.listdir(ops['data_path']) if '.mat' in f]
    if len(bpod_mat) == 1:
        shutil.copyfile(os.path.join(args.data_path, bpod_mat[0]),
            os.path.join(args.save_path, 'bpod_session_data.mat'))
    else:
        print('Valid bpod session data mat file not found')


# run with command line.

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experiments can go shit but Yicong will love you forever!')
    parser.add_argument('--denoise',         required=True, type=int, help='Whether run denoising.')
    parser.add_argument('--spatial_scale',   required=True, type=int, help='The optimal scale in suite2p.')
    parser.add_argument('--data_path',       required=True, type=str, help='Path to the 2P imaging data.')
    parser.add_argument('--save_path',       required=True, type=str, help='Path to save the results.')
    parser.add_argument('--nchannels',       required=True, type=int, help='Specify the number of channels.')
    parser.add_argument('--functional_chan', required=True, type=int, help='Specify functional channel id.')
    parser.add_argument('--brain_region',    required=True, type=str, help='Can only be crbl or ppc.')
    args = parser.parse_args()

    ops, db = set_params(args)

    process_vol(args)

    move_bpod_mat(args)

    run_s2p(ops=ops, db=db)

