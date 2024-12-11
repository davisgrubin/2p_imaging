#!/usr/bin/env python3

import os
import h5py
import tifffile
import numpy as np
from tqdm import tqdm
from cellpose import models
from cellpose import io


# z score normalization.

def normz(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-5)


# run cellpose on one image for cell detection and save the results.

def run_cellpose(
        ops, mean_anat,
        diameter,
        flow_threshold=0.5,
        ):
    if not os.path.exists(os.path.join(ops['save_path0'], 'cellpose')):
        os.makedirs(os.path.join(ops['save_path0'], 'cellpose'))
    tifffile.imwrite(
        os.path.join(ops['save_path0'], 'cellpose', 'mean_anat.tif'),
        mean_anat)
    model = models.Cellpose(model_type="cyto3")
    masks_anat, flows, styles, diams = model.eval(
        mean_anat,
        diameter=diameter,
        flow_threshold=flow_threshold)
    io.masks_flows_to_seg(
        images=mean_anat,
        masks=masks_anat,
        flows=flows,
        file_names=os.path.join(ops['save_path0'], 'cellpose', 'mean_anat'),
        diams=diameter)
    return masks_anat


# read and cut mask in ops.

def get_mask(ops):
    masks_npy = np.load(
        os.path.join(ops['save_path0'], 'qc_results', 'masks.npy'),
        allow_pickle=True)
    x1 = ops['xrange'][0]
    x2 = ops['xrange'][1]
    y1 = ops['yrange'][0]
    y2 = ops['yrange'][1]
    masks_func = masks_npy[y1:y2,x1:x2]
    mean_func = ops['meanImg'][y1:y2,x1:x2]
    max_func = ops['max_proj']
    if ops['nchannels'] == 2:
        mean_anat = ops['meanImg_chan2'][y1:y2,x1:x2]
    else:
        mean_anat = None
    return masks_func, mean_func, max_func, mean_anat


# compute overlapping to get labels.

def get_label(
        masks_func, masks_anat,
        thres1=0.3, thres2=0.5,
        ):
    # reconstruct masks into 3d array.
    anat_roi_ids = np.unique(masks_anat)[1:]
    masks_3d = np.zeros((len(anat_roi_ids), masks_anat.shape[0], masks_anat.shape[1]))
    for i, roi_id in enumerate(anat_roi_ids):
        masks_3d[i] = (masks_anat == roi_id).astype(int)
    masks_3d[masks_3d!=0] = 1
    # compute relative overlaps coefficient for each functional roi.
    prob = []
    for i in tqdm(np.unique(masks_func)[1:]):
        roi_masks = (masks_func==i).astype('int32')
        roi_masks_tile = np.tile(
            np.expand_dims(roi_masks, 0),
            (len(anat_roi_ids),1,1))
        overlap = (roi_masks_tile * masks_3d).reshape(len(anat_roi_ids),-1)
        overlap = np.sum(overlap, axis=1)
        prob.append(np.max(overlap) / np.sum(roi_masks))
    # threshold probability to get label.
    prob = np.array(prob)
    labels = np.zeros_like(prob)
    # excitory.
    labels[prob < thres1] = -1
    # inhibitory.
    labels[prob > thres2] = 1
    return labels


# save channel img and masks results.

def save_masks(ops, masks_func, masks_anat, mean_func, max_func, mean_anat, labels):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'masks.h5'),
        'w')
    f['labels'] = labels
    f['masks_func'] = masks_func
    f['mean_func'] = mean_func
    f['max_func'] = max_func
    if ops['nchannels'] == 2:
        f['mean_anat'] = mean_anat
        f['masks_anat'] = masks_anat
    f.close()


# main function to use anatomical to label functional channel masks.

def run(ops, diameter):
    print('===============================================')
    print('===== two channel data roi identification =====')
    print('===============================================')
    print('Reading masks in functional channel')
    [masks_func, mean_func, max_func, mean_anat] = get_mask(ops)
    if np.max(masks_func) == 0:
        raise ValueError('No masks found.')
    if ops['nchannels'] == 1:
        print('Single channel recording so skip ROI labeling')
        labels = -1 * np.ones(int(np.max(masks_func))).astype('int32')
        save_masks(ops, masks_func, None, mean_func, max_func, None, labels)
    else:
        print('Running cellpose on anatomical channel mean image')
        print('Found diameter as {}'.format(diameter))
        masks_anat = run_cellpose(ops, mean_anat, diameter)
        print('Computing labels for each ROI')
        labels = get_label(masks_func, masks_anat)
        print('Found {} labeled ROIs out of {} in total'.format(
            np.sum(labels==1), len(labels)))
        save_masks(
            ops,
            masks_func, masks_anat, mean_func,
            max_func, mean_anat,
            labels)
        print('Masks results saved')
