#!/usr/bin/env python3

import gc
import os
import numpy as np
from datetime import datetime
from suite2p import extraction_wrapper


# extract fluorescence signals from masks given as stat file.

def get_fluorescence(
        ops,
        stat,
        f_reg_ch1,
        f_reg_ch2
        ):
    stat, F_ch1, Fneu_ch1, F_ch2, Fneu_ch2 = extraction_wrapper(
        stat = stat,
        f_reg = f_reg_ch1,
        f_reg_chan2 = f_reg_ch2,
        ops = ops)
    F = [F_ch1, F_ch2][ops['functional_chan']-1]
    Fneu = [Fneu_ch1, Fneu_ch2][ops['functional_chan']-1]
    return [stat, F, Fneu]


# save the pipeline data into .npy files.

def save_results(
        ops,
        stat,
        F, Fneu,
        ):
    np.save(os.path.join(ops['save_path0'], 'raw', 'stat.npy'), stat)
    np.save(os.path.join(ops['save_path0'], 'raw', 'F.npy'), F)
    np.save(os.path.join(ops['save_path0'], 'raw', 'Fneu.npy'), Fneu)


# delete registration binary files.

def clean_reg_bin(
        ops,
        f_reg_ch1, f_reg_ch2
        ):
    try:
        del f_reg_ch1
        del f_reg_ch2
        gc.collect()
        os.remove(os.path.join(ops['save_path0'], 'temp', 'reg_ch1.bin'))
        os.remove(os.path.join(ops['save_path0'], 'temp', 'reg_ch2.bin'))
    except:
        None
        

# main function for fluorescence signal extraction from ROIs.

def run(ops, stat, f_reg_ch1, f_reg_ch2):
    print('===============================================')
    print('======= extracting fluorescence signals =======')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    [stat, F, Fneu] = get_fluorescence(
         ops,
         stat,
         f_reg_ch1,
         f_reg_ch2)
    print('Fluorescence extraction completed')

    save_results(ops, stat, F, Fneu)
    print('Masks and traces data saved')

    clean_reg_bin(ops, f_reg_ch1, f_reg_ch2)

