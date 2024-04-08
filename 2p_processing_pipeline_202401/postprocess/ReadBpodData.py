#!/usr/bin/env python3

import numpy as np
import scipy.io as sio


# read bpod session data.

def read_bpod_mat_data(fname):
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], sio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d
    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d
    def _tolist(ndarray):
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    bpod_sess_data = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    bpod_sess_data = _check_keys(bpod_sess_data)
    bpod_sess_data = bpod_sess_data['SessionData']
    return bpod_sess_data



