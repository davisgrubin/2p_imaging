#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import multivariate_normal


# z score normalization.

def normz(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-5)


# signal ratio between ROI and its surroundings.

def get_roi_sur_ratio(
        mask_roi,
        anat_img
        ):
    mask_roi_ext = binary_dilation(mask_roi, iterations=2)
    # dilation for mask including surrounding.
    mask_roi_surr = binary_dilation(mask_roi_ext, iterations=5)
    # binary operation to get surrounding mask.
    mask_surr = mask_roi_surr != mask_roi_ext
    # compute mean signal.
    sig_roi = (anat_img * mask_roi_ext).reshape(-1)
    sig_roi = np.mean(sig_roi[sig_roi!=0])
    sig_surr = (anat_img * mask_surr).reshape(-1)
    sig_surr = np.mean(sig_surr[sig_surr!=0])
    # compute ratio.
    ratio = sig_roi - sig_surr
    return ratio


# compute spatial magnitude in anatomical channel.

def get_spat_mag(
        mask_roi,
        anat_img
        ):
    anat_roi = (anat_img * mask_roi).reshape(-1)
    anat_roi = anat_roi[anat_roi!=0].reshape(1,-1)
    mag = np.mean(anat_roi)
    return mag


# save channel img and masks results.

def save_masks(ops, masks, labels):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'masks.h5'),
        'w')
    x1 = ops['xrange'][0]
    x2 = ops['xrange'][1]
    y1 = ops['yrange'][0]
    y2 = ops['yrange'][1]
    f['labels'] = labels
    f['masks'] = masks[y1:y2,x1:x2]
    f['mean_func'] = ops['meanImg'][y1:y2,x1:x2]
    f['max_func'] = ops['max_proj']
    f['ref_img'] = ops['refImg'][y1:y2,x1:x2]
    if ops['nchannels'] == 2:
        f['mean_anat'] = ops['meanImg_chan2'][y1:y2,x1:x2]
    f.close()


# outlier detection for standard multivariate normal.

def outlier_detect(features):
    mean = np.mean(features, axis=0)
    cov = np.cov(features.T)
    model = multivariate_normal(mean=mean, cov=cov)
    prob = model.pdf(features)
    thres = model.pdf(mean + 1.5 * np.sqrt(np.diag(cov)))
    outlier = prob < thres
    return outlier


# main function to use anatomical to label functional channel masks.

def run(ops):
    print('===============================================')
    print('===== two channel data roi identification =====')
    print('===============================================')
    masks = np.load(
        os.path.join(ops['save_path0'], 'qc_results', 'masks.npy'),
        allow_pickle=True)
    if np.max(masks) == 0:
        raise ValueError('No masks found.')
    if ops['nchannels'] == 1:
        print('Single channel recording so skip ROI labeling')
        labels = np.zeros(int(np.max(masks))).astype('int32')
        save_masks(ops, masks, labels)
    else:
        # specify functional and anatomical masks.
        #func_img = ops['meanImg']
        anat_img = ops['meanImg_chan2']
        surr_ratio = []
        spat_mag = []
        print('Collecting ROI statistics')
        for i in np.unique(masks)[1:]:
            # get ROI mask.
            mask_roi = (masks==i).copy()
            # get signal ratio between roi and its surroundings.
            r = get_roi_sur_ratio(mask_roi, anat_img)
            # get spatial correlation around roi.
            m = get_spat_mag(mask_roi, anat_img)
            # collect results.
            surr_ratio.append(r)
            spat_mag.append(m)
        features = np.concatenate(
            (normz(spat_mag).reshape(-1,1),
             normz(surr_ratio).reshape(-1,1)
             ), axis=1)
        # import matplotlib.pyplot as plt
        # plt.hist(spat_mag, bins=50)
        # plt.hist(surr_ratio, bins=50)
        # plt.scatter(features[:,0], features[:,1])
        # 0 : only in functional.
        # 1 : in functional and marked in anatomical.
        print('Running outlier detection')
        outlier = outlier_detect(features)
        labels = outlier.astype('int32')
        print('Found {} inhibitory neurons with new ROI id {}'.format(
            len(np.where(labels)[0]), np.where(labels)[0]))
        save_masks(ops, masks, labels)
