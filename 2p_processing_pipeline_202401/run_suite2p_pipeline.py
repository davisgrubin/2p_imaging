#!/usr/bin/env python3
import os
import json
import h5py
import argparse
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
        # column 0: time index in ms.
        # column 1: trial start signal from bpod.
        # column 2: stimulus signal from photodiode.
        # column 3: BNC2 not in use.
        # column 4: image trigger signal from 2p scope camera.
        vol_time  = df_vol.iloc[:,0].to_numpy()
        vol_start = df_vol.iloc[:,1].to_numpy()
        vol_stim  = df_vol.iloc[:,2].to_numpy()
        vol_img   = df_vol.iloc[:,4].to_numpy()
        return vol_time, vol_start, vol_stim, vol_img

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
    def vol_to_binary(
            vol_start,
            vol_stim,
            vol_img
            ):
        vol_start_bin = thres_binary(vol_start, 1)
        vol_stim_bin  = thres_binary(vol_stim, 1)
        vol_img_bin   = thres_binary(vol_img, 1)
        return vol_start_bin, vol_stim_bin, vol_img_bin

    # save voltage data.
    def save_vol(
            args,
            vol_time, vol_start_bin, vol_stim_bin, vol_img_bin,
            ):
        # file structure:
        # args.save_path / raw_voltages.h5
        # -- raw
        # ---- vol_time
        # ---- vol_start_bin
        # ---- vol_stim_bin
        # ---- vol_img_bin
        f = h5py.File(os.path.join(
            args.save_path, 'raw_voltages.h5'), 'w')
        grp = f.create_group('raw')
        grp['vol_time']      = vol_time
        grp['vol_start_bin'] = vol_start_bin
        grp['vol_stim_bin']  = vol_stim_bin
        grp['vol_img_bin']   = vol_img_bin
        f.close()

    # run processing.
    try:
        vol_time, vol_start, vol_stim, vol_img = read_vol_to_np(args)
        vol_start_bin, vol_stim_bin, vol_img_bin = vol_to_binary(
            vol_start, vol_stim, vol_img)
        save_vol(args, vol_time, vol_start_bin, vol_stim_bin, vol_img_bin)
    except:
        print('Valid voltage recordings csv file not found')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experiments can go shit but Yicong will love you forever!')
    parser.add_argument('--denoise',         required=True, type=int, help='Whether run denoising algorithm.')
    parser.add_argument('--spatial_scale',   required=True, type=int, help='The optimal scale in suite2p.')
    parser.add_argument('--data_path',       required=True, type=str, help='Path to the 2P imaging data.')
    parser.add_argument('--save_path',       required=True, type=str, help='Path to save the results.')
    parser.add_argument('--nchannels',       required=True, type=int, help='Specify the number of channels.')
    parser.add_argument('--functional_chan', required=True, type=int, help='Specify functional channel id.')
    parser.add_argument('--brain_region',    required=True, type=str, help='Can only be crbl or ppc.')
    args = parser.parse_args()

    ops, db = set_params(args)

    process_vol(args)

    run_s2p(ops=ops, db=db)

