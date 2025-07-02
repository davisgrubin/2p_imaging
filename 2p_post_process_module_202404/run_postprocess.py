#!/usr/bin/env python3

import os
import argparse
import numpy as np

from modules import QualControlDataIO
from modules import LabelExcInh
from modules import DffTraces
COMMANDLINE_MODE = False

# parse arg input to number list.
def get_qc_args(args):
    range_skew = args.range_skew.split(',')
    range_skew = np.array(range_skew, dtype='float32')
    max_connect = np.array(args.max_connect, dtype='float32')
    range_aspect = args.range_aspect.split(',')
    range_aspect = np.array(range_aspect, dtype='float32')
    range_compact = args.range_compact.split(',')
    range_compact = np.array(range_compact, dtype='float32')
    range_footprint = args.range_footprint.split(',')
    range_footprint = np.array(range_footprint, dtype='float32')
    return [range_skew,
            max_connect,
            range_aspect,
            range_compact,
            range_footprint]

# read saved ops.npy given a folder in ./results.
def read_ops(session_data_path):
    print('Processing {}'.format(session_data_path))
    ops = np.load(
        os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session_data_path)
    return ops

if __name__ == "__main__":
    import sys
    print(sys.argv)
    COMMANDLINE_MODE = True
    if COMMANDLINE_MODE:
        parser = argparse.ArgumentParser(description='Do not forget the everlasting love from Yicong!')
        parser.add_argument('--session_data_path', required=True, type=str, help='The name of folder to save suite2p results.')
        parser.add_argument('--range_skew',        required=True, type=str, help='The range of skew for quality control.')
        parser.add_argument('--max_connect',       required=True, type=str, help='The maximum number of connectivity for quality control.')
        parser.add_argument('--range_footprint',   required=True, type=str, help='The range of footprint for quality control')
        parser.add_argument('--range_aspect',      required=True, type=str, help='The range of aspect ratio for quality control.')
        parser.add_argument('--range_compact',     required=True, type=str, help='The range of compact for quality control.')
        parser.add_argument('--diameter',          required=True, type=str, help='The diameter for cellpose on anatomical channel.')
        args = parser.parse_args()
        [range_skew,
         max_connect,
         range_aspect,
         range_compact,
         range_footprint] = get_qc_args(args)
        ops = read_ops(args.session_data_path)
        session_data_path_list = args.session_data_path.split(',')
    else:
        '''
        session_data_path_list = [
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250106_3331Random',
        '''
        session_data_path_list = [
            'C:/Users/yhuang887/Projects/interval_discrimination_202501/single_interval/results/YH24LG/YH24LG_CRBL_crux1_20250427_2afc', 'C:/Users/yhuang887/Projects/interval_discrimination_202501/single_interval/results/YH24LG/YH24LG_CRBL_crux1_20250428_2afc', 'C:/Users/yhuang887/Projects/interval_discrimination_202501/single_interval/results/YH24LG/YH24LG_CRBL_crux1_20250429_2afc', 'C:/Users/yhuang887/Projects/interval_discrimination_202501/single_interval/results/YH24LG/YH24LG_CRBL_crux1_20250425_2afc', 'C:/Users/yhuang887/Projects/interval_discrimination_202501/single_interval/results/YH24LG/YH24LG_CRBL_crux1_20250426_2afc'
            ]

        
            
        # dendrites.
        range_skew = [0,2]
        max_connect = 2
        range_aspect = [1.2,5]
        range_footprint = [1,2]
        range_compact = [1.06,5]
        diameter = 6
        '''
        # neurons.
        range_skew = [-5,5]
        max_connect = 1
        range_aspect = [0,5]
        range_footprint = [1,2]
        range_compact = [0,1.06]
        diameter = 6
        '''
    for session_data_path in session_data_path_list:
        ops = read_ops(session_data_path)
        QualControlDataIO.run(
                ops,
                range_skew, max_connect, range_aspect, range_compact, range_footprint)
        LabelExcInh.run(ops, diameter)
        DffTraces.run(ops, correct_pmt=False)
        print('Finished Processing {}'.format(session_data_path))
