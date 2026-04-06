#!/usr/bin/env python3

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from modules import QualControlDataIO
from modules import LabelExcInh
from modules import DffTraces
from modules.ReadResults import read_masks
from modules.fig1_mask import plotter_all_masks
COMMANDLINE_MODE = False


# parse arg input to number list.

def get_qc_args(args):
    range_skew = args.range_skew.split(',')
    range_skew = np.array(range_skew, dtype='float32')
    max_connect = np.array(args.max_connect, dtype='float32')
    max_aspect = np.array(args.max_aspect, dtype='float32')
    range_compact = args.range_compact.split(',')
    range_compact = np.array(range_compact, dtype='float32')
    range_footprint = args.range_footprint.split(',')
    range_footprint = np.array(range_footprint, dtype='float32')
    return [range_skew,
            max_connect,
            max_aspect,
            range_compact,
            range_footprint]


# read saved ops.npy given a folder in ./results.

def read_ops(session_data_path):
    print('Processing {}'.format(session_data_path))
    ops = np.load(
        os.path.join(session_data_path, 'suite2p', 'plane0', 'ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session_data_path)
    return ops


def run(session_data_path):

    if COMMANDLINE_MODE:
        parser = argparse.ArgumentParser(
            description='Do not forget the everlasting love from Yicong!')
        parser.add_argument('--session_data_path', required=True,
                            type=str, help='The name of folder to save suite2p results.')
        parser.add_argument('--range_skew',        required=True,
                            type=str, help='The range of skew for quality control.')
        parser.add_argument('--max_connect',       required=True, type=str,
                            help='The maximum number of connectivity for quality control.')
        parser.add_argument('--range_footprint',   required=True,
                            type=str, help='The range of footprint for quality control')
        parser.add_argument('--max_aspect',        required=True, type=str,
                            help='The maximum value of aspect ratio for quality control.')
        parser.add_argument('--range_compact',     required=True,
                            type=str, help='The range of compact for quality control.')
        parser.add_argument('--diameter',          required=True, type=str,
                            help='The diameter for cellpose on anatomical channel.')
        args = parser.parse_args()
        [range_skew,
         max_connect,
         max_aspect,
         range_compact,
         range_footprint] = get_qc_args(args)
        ops = read_ops(args.session_data_path)
    else:
        ops = read_ops(session_data_path)
        '''
        range_skew = [1,2]
        max_connect = 5
        max_aspect = 55
        range_footprint = [1,2]
        range_compact = [1.2,5]
        diameter = 6
        '''
        # range_skew = [0.56,5]
        # max_connect = 10
        # max_aspect = 55
        # range_footprint = [1,2]
        # range_compact = [1.3, 5]
        # diameter = 6
        
        # range_skew = [-1,5]
        # max_connect = 60
        # max_aspect = 55
        # range_footprint = [1,2]
        # range_compact = [1, 5]
        # diameter = 6
        
        range_skew = [0.4,1]
        max_connect = 30
        max_aspect = 55
        range_footprint = [1,2]
        range_compact = [1, 1.5]
        diameter = 6
    '''
    QualControlDataIO.run(
        ops, range_skew, max_connect, max_aspect, range_compact, range_footprint,
        run_qc=False)
    '''

    QualControlDataIO.run(
        ops,
        range_skew, max_connect, max_aspect, range_compact, range_footprint, stat_file_names=['stat'])

    LabelExcInh.run(ops, diameter)

    DffTraces.run(ops, taus=[1.25])
    
    print('================plot the FOV==============')

    [labels, masks, mean_func, max_func, mean_anat, masks_anat, masks_anat_corrected] = read_masks(ops)
    plotter_masks = plotter_all_masks(
            labels, masks, mean_func, max_func, mean_anat, masks_anat)
    fig = plt.figure(figsize=(105, 210))

    gs = GridSpec(30, 15, figure=fig)
    mask_ax01 = plt.subplot(gs[0:2, 0:2])
    mask_ax02 = plt.subplot(gs[0:2, 2:4])
    plotter_masks.func(mask_ax01, 'max')
    plotter_masks.func_masks(mask_ax02)

    plt.show()
    fname = session_data_path + '/qc_results/' + 'mask.pdf'
    fig.savefig(fname, transparent=True, bbox_inches='tight',dpi=300)