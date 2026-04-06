# -*- coding: utf-8 -*-
"""
Created on Tue May 20 17:39:06 2025

@author: saminnaji3
"""


import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from modules import QualControlDataIO
from modules import LabelExcInh
from modules import DffTraces
from modules.ReadResults import read_masks
from modules.fig1_mask import plotter_all_masks

COMMANDLINE_MODE = False
manual_quality_control = True


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
    
    diameter = 6
    
    if manual_quality_control:
        ops = read_ops(session_data_path)
        QualControlDataIO.run_manual(
            ops, stat_file_names=['stat'])
    
        LabelExcInh.run_manual(ops, diameter)
    
        DffTraces.run_manual(ops, taus=[1.25])
        
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
        fname = session_data_path + '/manual_qc_results/' + 'mask.pdf'
        fig.savefig(fname, transparent=True, bbox_inches='tight',dpi=300)