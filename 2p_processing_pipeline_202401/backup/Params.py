#!/usr/bin/env python3

import os
import json
from datetime import datetime


# main function for setting the ops.npy for suite2p.

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
    if not os.path.exists(os.path.join(args.save_path, 'raw')):
        os.makedirs(os.path.join(args.save_path, 'raw'))
    if not os.path.exists(os.path.join(args.save_path, 'figures')):
        os.makedirs(os.path.join(args.save_path, 'figures'))
        
    # set params specified by command line.
    ops['Lx']              = 512
    ops['Ly']              = 512
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

    # save ops.npy to the path.
    print('Parameters setup for {} completed'.format(args.brain_region))

    return ops

