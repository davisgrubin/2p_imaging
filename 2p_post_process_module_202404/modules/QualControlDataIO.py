#!/usr/bin/env python3

import os
import h5py
import numpy as np
from skimage.measure import label


# read raw results from suite2p pipeline.

def read_raw(ops):
    F = np.load(
        os.path.join(ops['save_path0'],
        'suite2p', 'plane0', 'F.npy'), allow_pickle=True)
    Fneu = np.load(
        os.path.join(ops['save_path0'],
        'suite2p', 'plane0', 'Fneu.npy'), allow_pickle=True)
    stat = np.load(
        os.path.join(ops['save_path0'],
        'suite2p', 'plane0', 'stat.npy'), allow_pickle=True)
    return [F, Fneu, stat]


# get metrics for ROIs.

def get_metrics(ops, stat):
    # rearrange existing statistics for masks.
    # https://suite2p.readthedocs.io/en/latest/outputs.html#stat-npy-fields
    footprint = np.array([stat[i]['footprint']    for i in range(len(stat))])
    skew      = np.array([stat[i]['skew']         for i in range(len(stat))])
    aspect    = np.array([stat[i]['aspect_ratio'] for i in range(len(stat))])
    compact   = np.array([stat[i]['compact'] for i in range(len(stat))])
    # compute connetivity of ROIs.
    masks = stat_to_masks(ops, stat)
    connect = []
    for i in np.unique(masks)[1:]:
        # find a mask with one roi.
        m = masks.copy() * (masks == i)
        # find component number.
        connect.append(np.max(label(m, connectivity=1)))
    connect = np.array(connect)
    return skew, connect, aspect, compact, footprint


# threshold the statistics to keep good ROIs.

def thres_stat(
        ops, stat,
        range_skew,
        max_connect,
        max_aspect,
        range_compact,
        range_footprint
        ):
    skew, connect, aspect, compact, footprint = get_metrics(ops, stat)
    # find bad roi indice.
    bad_roi_id = set()
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where((footprint<range_footprint[0]) | (footprint>range_footprint[1]))[0])
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where((skew<range_skew[0]) | (skew>range_skew[1]))[0])
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where(aspect>max_aspect)[0])
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where((compact<range_compact[0]) | (compact>range_compact[1]))[0])
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where(connect>max_connect)[0])
    # convert set to numpy array for indexing.
    bad_roi_id = np.array(list(bad_roi_id))
    return bad_roi_id


# reset bad ROIs in the masks to nothing.

def reset_roi(
        bad_roi_id,
        F, Fneu, stat
        ):
    # reset bad roi.
    for i in bad_roi_id:
        stat[i]   = None
        F[i,:]    = 0
        Fneu[i,:] = 0
    # find good roi indice.
    good_roi_id = np.where(np.sum(F, axis=1)!=0)[0]
    # keep good roi signals.
    fluo = F[good_roi_id,:]
    neuropil = Fneu[good_roi_id,:]
    stat = stat[good_roi_id]
    return fluo, neuropil, stat


# save results into npy files.

def save_qc_results(
        ops,
        fluo, neuropil, stat, masks
        ):
    if not os.path.exists(os.path.join(ops['save_path0'], 'qc_results')):
        os.makedirs(os.path.join(ops['save_path0'], 'qc_results'))
    np.save(os.path.join(ops['save_path0'], 'qc_results', 'fluo.npy'), fluo)
    np.save(os.path.join(ops['save_path0'], 'qc_results', 'neuropil.npy'), neuropil)
    np.save(os.path.join(ops['save_path0'], 'qc_results', 'stat.npy'), stat)
    np.save(os.path.join(ops['save_path0'], 'qc_results', 'masks.npy'), masks)
    np.save(os.path.join(ops['save_path0'], 'ops.npy'), ops)


# convert stat.npy results to ROI masks matrix.

def stat_to_masks(ops, stat):
    masks = np.zeros((ops['Ly'], ops['Lx']))
    for n in range(len(stat)):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        masks[ypix,xpix] = n+1
    return masks


# save motion correction offsets.

def save_move_offset(ops):
    xoff = ops['xoff']
    yoff = ops['yoff']
    f = h5py.File(os.path.join(ops['save_path0'], 'move_offset.h5'), 'w')
    f['xoff'] = xoff
    f['yoff'] = yoff
    f.close()


# main function for quality control.

def run(
        ops,
        range_skew, max_connect, max_aspect, range_compact, range_footprint,
        run_qc=True
        ):
    print('===============================================')
    print('=============== quality control ===============')
    print('===============================================')
    print('Found range of footprint from {} to {}'.format(
        range_footprint[0], range_footprint[1]))
    print('Found range of skew from {} to {}'.format(
        range_skew[0], range_skew[1]))
    print('Found max number of connectivity components as {}'.format(
        max_connect))
    print('Found maximum aspect ratio as {}'.format(
        max_aspect))
    print('Found range of campact as {}'.format(
        range_compact))
    [F, Fneu, stat] = read_raw(ops)
    print('Found {} ROIs from suite2p'.format(F.shape[0]))
    if run_qc:
        bad_roi_id = thres_stat(
            ops, stat,
            range_skew, max_connect, max_aspect, range_compact, range_footprint)
        print('Found {} bad ROIs'.format(len(bad_roi_id)))
    else:
        bad_roi_id = []
    fluo, neuropil, stat = reset_roi(bad_roi_id, F, Fneu, stat)
    print('Saving {} ROIs after quality control'.format(fluo.shape[0]))
    masks = stat_to_masks(ops, stat)
    save_qc_results(ops, fluo, neuropil, stat, masks)
    save_move_offset(ops)


