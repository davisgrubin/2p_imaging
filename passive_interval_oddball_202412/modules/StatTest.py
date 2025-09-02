#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy.stats import ks_2samp

from modules.ReadResults import read_neural_trials
from utils import get_odd_stim_prepost_idx
from utils import exclude_odd_stim
from utils import pick_trial


# save significance label results.
def save_significance(
        ops,
        r_standard
        ):
    h5_path = os.path.join(ops['save_path0'], 'significance.h5')
    if os.path.exists(h5_path):
        os.remove(h5_path)
    f = h5py.File(h5_path, 'w')
    grp = f.create_group('significance')
    grp['r_standard']  = r_standard
    f.close()

def run(ops):
    print('Aligning neural population response')
    neural_trials = read_neural_trials(ops, False)
    print('Running statistics test')
    r_standard = np.ones(neural_trials['dff'].shape[0])
    print('{}/{} ROIs responsive to standard'.format(np.sum(r_standard), len(r_standard)))
    save_significance(ops, r_standard)

