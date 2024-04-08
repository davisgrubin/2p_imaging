#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import binary_dilation


# signal ratio between ROI and its surroundings.

def get_roi_sur_ratio(
        mask_roi,
        anat_img
        ):
    # dilation for mask including surrounding.
    mask_roi_surr = binary_dilation(mask_roi, iterations=5)
    # binary operation to get surrounding mask.
    mask_surr = mask_roi_surr != mask_roi
    # compute mean signal.
    sig_roi = (anat_img * mask_roi).reshape(-1)
    sig_roi = np.mean(sig_roi[sig_roi!=0])
    sig_surr = (anat_img * mask_surr).reshape(-1)
    sig_surr = np.mean(sig_surr[sig_surr!=0])
    # compute ratio.
    ratio = sig_roi / (sig_surr + 1e-10)
    return ratio


# compute spatial correlation.

def get_spat_corr(
        mask_roi,
        func_img, anat_img
        ):
    func_roi = (func_img * mask_roi).reshape(-1)
    func_roi = func_roi[func_roi!=0].reshape(1,-1)
    anat_roi = (anat_img * mask_roi).reshape(-1)
    anat_roi = anat_roi[anat_roi!=0].reshape(1,-1)
    corr = np.corrcoef(func_roi, anat_roi)
    corr = corr[0,1]
    return corr


# main function to use anatomical to label functional channel masks.

def run(ops, masks):
    if np.max(masks) == 0:
        raise ValueError('No masks found.')
    if ops['nchannels'] == 1:
        labels = np.zeros(int(np.max(masks)))
        return labels
    else:
        # specify functional and anatomical masks.
        func_img = ops['meanImg']
        anat_img = ops['meanImg_chan2']
        surr_ratio = []
        corr = []
        for i in np.unique(masks)[1:]:
            # get ROI mask.
            mask_roi = (masks==i).copy()
            # get signal ratio between roi and its surroundings.
            r = get_roi_sur_ratio(mask_roi, anat_img)
            # compute spatial correlation.
            c = get_spat_corr(mask_roi, func_img, anat_img)
            # collect results.
            surr_ratio.append(r)
            corr.append(c)
        # 0 : only in functional.
        # 1 : in functional and marked in anatomical.
        label = (surr_ratio > (np.mean(surr_ratio)+3*np.std(surr_ratio)))*1
        return label
