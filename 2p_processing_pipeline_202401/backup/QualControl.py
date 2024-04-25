#!/usr/bin/env python3

import os
import h5py
import numpy as np


# read raw results from suite2p pipeline

def read_raw(ops):
    # read raw traces.
    f = h5py.File(
        os.path.join(ops['save_path0'], 'raw', 'raw_traces.h5'),
        'r')
    fluo = np.array(f['raw']['fluo'])
    neuropil = np.array(f['raw']['neuropil'])
    f.close()
    # read raw masks stat.
    stat = np.load(
        os.path.join(ops['save_path0'], 'raw', 'stat.npy'),
        allow_pickle=True)
    # rearrange statistics for masks.
    # https://suite2p.readthedocs.io/en/latest/outputs.html#stat-npy-fields
    footprint = np.array([stat[i]['footprint']    for i in range(len(stat))])
    skew      = np.array([stat[i]['skew']         for i in range(len(stat))])
    npix_norm = np.array([stat[i]['npix_norm'] for i in range(len(stat))])
    mrs       = np.array([stat[i]['mrs'] for i in range(len(stat))])
    return [fluo, neuropil,
            footprint, skew, npix_norm, mrs]


# threshold the statistics to keep good ROIs.

def thres_stat(
        footprint,
        skew,
        mrs,
        ):
    bad_roi_id = set()
    # spatial extent is abnormal.
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where((footprint==0) | (footprint==3))[0])
    # skewness is too low so almost all noise.
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where(skew < np.mean(skew)+np.std(skew))[0])
    # too large on some directions.
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where(mrs > np.mean(mrs) + 2*np.std(mrs))[0])
    # convert set to numpy array for indexing.
    bad_roi_id = np.array(list(bad_roi_id))
    print(len(bad_roi_id))


# reset bad ROIs in the masks to nothing.

def reset_masks():
    0
'''
masks_sqeeze = masks.copy().reshape(-1)
for i in bad_roi_id:
    masks_sqeeze[masks_sqeeze==i] = 0
masks_qc = masks_sqeeze.reshape(ops['Ly'], ops['Lx'])
plt.imshow(masks, 'random')
plt.imshow(masks_qc, 'inferno')
np.where(fluo==np.max(fluo))
plt.hist(np.mean(fluo,axis=1), bins=100)

plt.hist(skew, bins=100)
'''


