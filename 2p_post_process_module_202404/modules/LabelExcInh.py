#!/usr/bin/env python3

import os
import h5py
import tifffile
import numpy as np
from scipy.optimize import minimize
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

# read the trace and compute fluorescence traces.
def get_ch_traces(ops):
    fluo_ch1 = np.load(os.path.join(ops['save_path0'], 'suite2p', 'plane0', 'F.npy'), allow_pickle=True)
    fluo_ch2 = np.load(os.path.join(ops['save_path0'], 'suite2p', 'plane0', 'F_chan2.npy'), allow_pickle=True)
    return fluo_ch1, fluo_ch2

# bleedthrough correction for red channel mean projection.
def anat_bleedthrough_correction(mean_anat, mean_func, fluo_ch1, fluo_ch2):
    # objective function to optimize.
    def objective(params, fluo_ch1, fluo_ch2):
        a, b = params
        # linear correction.
        fluo_ch1_corrected = fluo_ch1 - (fluo_ch2 * a + b)
        # correlation between corrected anantomical channel with functional channel.
        c = [np.corrcoef(fluo_ch1_corrected[i], fluo_ch2[i])[0, 1] for i in range(fluo_ch1.shape[0])]
        c = np.mean(np.abs(c))
        return c
    # find optimal parameters.
    def optimize():
        val_params = []
        val_objective = []
        init = [0, 0]
        result = minimize(
            objective,
            init,
            args=(fluo_ch1, fluo_ch2),
            method='L-BFGS-B',
            callback= lambda xk:
                (val_params.append(xk.copy()),
                 val_objective.append(objective(xk, fluo_ch1, fluo_ch2))))
        a_opt, b_opt = result.x
        return a_opt, b_opt
    # mean projection correction.
    def correct_anat(a_opt, b_opt):
        mean_anat_corrected = mean_anat - (mean_func * a_opt + b_opt)
        return mean_anat_corrected
    # main.
    a_opt, b_opt = optimize()
    mean_anat_corrected = correct_anat(a_opt, b_opt)
    return mean_anat_corrected

# compute overlapping to get labels.
def get_label(
        masks_func, masks_anat,
        thres1=0.2, thres2=0.9,
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
        # extract masks with one roi.
        roi_masks_func = (masks_func==i).astype('int32')
        # expand roi to match the number of neurons in anatomical channel.
        roi_masks_tile = np.tile(
            np.expand_dims(roi_masks_func, 0),
            (len(anat_roi_ids),1,1))
        # compute all possible overlap.
        overlap = (roi_masks_tile * masks_3d).reshape(len(anat_roi_ids),-1)
        overlap = np.sum(overlap, axis=1)
        # find the roi masks in anatomical channel with highest overlap.
        roi_masks_anat = (masks_anat==(np.argmax(overlap)+1)).astype('int32')
        # find the maximum overlap.
        prob.append(np.max([np.max(overlap) / (np.sum(roi_masks_func)+1e-10),
                            np.max(overlap) / (np.sum(roi_masks_anat)+1e-10)]))
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
        print('Computing traces')
        fluo_ch1, fluo_ch2 = get_ch_traces(ops)
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

'''

fig, ax = plt.subplots(2, 3, figsize=(18, 12))
from scipy.ndimage import median_filter
[masks_func, mean_func, max_func, mean_anat] = get_mask(ops)

f = mean_func
func_img = np.zeros(
    (f.shape[0], f.shape[1], 3), dtype='int32')
func_img[:,:,1] = adjust_contrast(f)
func_img = adjust_contrast(func_img)
ax[0,0].matshow(func_img)
adjust_layout(ax[0,0])

f = max_func
func_img = np.zeros(
    (f.shape[0], f.shape[1], 3), dtype='int32')
func_img[:,:,1] = adjust_contrast(f)
func_img = adjust_contrast(func_img)
ax[0,1].matshow(func_img)
adjust_layout(ax[0,1])

f = median_filter(max_func, size=3)
func_img = np.zeros(
    (f.shape[0], f.shape[1], 3), dtype='int32')
func_img[:,:,1] = adjust_contrast(f)
func_img = adjust_contrast(func_img)
ax[0,2].matshow(func_img)
adjust_layout(ax[0,2])


mean_anat_corrected = anat_bleedthrough_correction(
    mean_anat, mean_func, fluo_ch1, fluo_ch2)

f = mean_anat
func_img = np.zeros(
    (f.shape[0], f.shape[1], 3), dtype='int32')
func_img[:,:,0] = adjust_contrast(f)
func_img = adjust_contrast(func_img)
ax[1,0].matshow(func_img)
adjust_layout(ax[1,0])

f = mean_anat_corrected
func_img = np.zeros(
    (f.shape[0], f.shape[1], 3), dtype='int32')
func_img[:,:,0] = adjust_contrast(f)
func_img = adjust_contrast(func_img)
ax[1,1].matshow(func_img)
adjust_layout(ax[1,1])

f = mean_anat - mean_anat_corrected
func_img = np.zeros(
    (f.shape[0], f.shape[1], 3), dtype='int32')
func_img[:,:,0] = adjust_contrast(f)
func_img = adjust_contrast(func_img)
ax[1,2].matshow(func_img)
adjust_layout(ax[1,2])

'''