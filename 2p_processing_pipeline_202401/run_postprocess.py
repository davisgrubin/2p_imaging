#!/usr/bin/env python3

import os
import argparse
import numpy as np

from postprocess import QualControlDataIO
from postprocess import LabelExcInh
from postprocess import DffTraces
from postprocess import Trialization
from postprocess import Visualization


'''
python run_postprocess.py `
--session_name 'FN15_P_omi_032124_w' `
--range_skew '0,5' `
--max_connect '3' `
--range_footprint '1,3' `
'''


# parse arg input to number list.

def get_qc_args(args):
    range_skew = args.range_skew.split(',')
    range_skew = np.array(range_skew, dtype='float32')
    max_connect = np.array(args.max_connect, dtype='float32')
    range_footprint = args.range_footprint.split(',')
    range_footprint = np.array(range_footprint, dtype='float32')
    return range_skew, max_connect, range_footprint


# read saved ops.npy given a folder in ./results.

def read_ops(session_name):
    ops = np.load(
        os.path.join('./results', session_name, 'suite2p', 'plane0','ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join('./results', session_name)
    return ops


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Do not forget the everlasting love from Yicong!')
    parser.add_argument('--session_name',    required=True, type=str, help='The name of folder to save suite2p results.')
    parser.add_argument('--range_skew',      required=True, type=str, help='The range of skew for quality control.')
    parser.add_argument('--max_connect',     required=True, type=str, help='The maximum number of connectivity for quality control.')
    parser.add_argument('--range_footprint', required=True, type=str, help='The range of footprint for quality control')
    args = parser.parse_args()

    '''
    session_name = 'FN15_P_omi_032124_w'
    ops = read_ops(session_name)
    range_skew = [0,5]
    max_connect = 3
    range_footprint = [1,3]
    '''

    range_skew, max_connect, range_footprint = get_qc_args(args)

    ops = read_ops(args.session_name)

    QualControlDataIO.run(
        ops, range_footprint, range_skew, max_connect)

    LabelExcInh.run(ops)

    DffTraces.run(ops)

    Trialization.run(ops)

    Visualization.run(ops)
