#!/usr/bin/env python3

import os
import numpy as np
from cellpose import models
from cellpose import io
from suite2p.detection import roi_stats


# identify functional and anatomical channels.

def get_channel_img(
        ops,
        proj_img,
        ch
        ):
    # functional channel.
    if ch == ops['functional_chan']:
        img = proj_img['ref_ch'+str(ops['functional_chan'])]
        return img
    # anatomical channel.
    if ch == 3-ops['functional_chan']:
        img = proj_img['ref_ch'+str(3-ops['functional_chan'])]
        return img


# run cellpose on one image for cell detection and save the results.

def run_cellpose(ops, img):
    # initialize cellpose pretrained model.
    model = models.Cellpose(model_type=ops['pretrained_model'])
    # run cellpose on the given image with the shape of Lx*Ly.
    masks, flows, styles, diams = model.eval(
        img,
        diameter=ops['diameter'] if ops['diameter'] != 0 else None,
        flow_threshold=ops['flow_threshold'],
        channels=[[0,0]],
        channel_axis=0)
    return [masks, flows, styles, diams]


# save cell segmentation to file XXX_seg.npy and reorganize it.

def save_cellpose_results(img, masks, flows, diams, file_names):
    io.masks_flows_to_seg(img, masks, flows, diams, file_names, [[0,0]])
    results = dict(
        input_img = img,
        masks = masks)
    return results
    

# identify two channels and get the best results.

def cellpose_eval(
        ops,
        proj_img,
        ch,
        file_names,
        ):
    # functional channel.
    if ch == ops['functional_chan']:
        # find the result with most neurons.
        results = []
        img = [proj_img['ref_ch'+str(ops['functional_chan'])],
               proj_img['max_ch'+str(ops['functional_chan'])]]
        results = [
            run_cellpose(ops, img[0]),
            run_cellpose(ops, img[1])]
        num = [len(np.unique(results[0][0]))-1,
               len(np.unique(results[1][0]))-1]
        masks, flows, styles, diams = results[np.argmax(num)]
        img = img[np.argmax(num)]
        results = save_cellpose_results(
            img, masks, flows, diams, file_names)
    # anatomical channel.
    if ch == 3-ops['functional_chan']:
        # use reference image.
        img = proj_img['ref_ch'+str(3-ops['functional_chan'])]
        masks, flows, styles, diams = run_cellpose(ops, img)
        results = save_cellpose_results(
            img, masks, flows, diams, file_names)
    return results


# run cellpose on mean channel image and reference image.

def get_mask(
        ops,
        proj_img
        ):
    # ch1 image.
    print('Running cellpose on ch1 image')
    results_ch1 = cellpose_eval(
        ops, proj_img, 1,
        os.path.join(ops['save_path0'], 'temp', 'mask_ch1'))
    # ch2 image.
    print('Running cellpose on ch2 image')
    results_ch2 = cellpose_eval(
        ops, proj_img, 2,
        os.path.join(ops['save_path0'], 'temp', 'mask_ch2'))
    return results_ch1, results_ch2


# reconstruct stat file for suite2p from mask.
# https://github.com/MouseLand/suite2p/issues/292.

def get_stat(
        ops,
        results_ch1,
        results_ch2,
        ):
    if ops['functional_chan'] == 1:
        masks = results_ch1['masks']
    if ops['functional_chan'] == 2:
        masks = results_ch2['masks']
    stat = []
    if np.max(masks) > 0:
        for u_ix, u in enumerate(np.unique(masks)[1:]):
            ypix, xpix = np.nonzero(masks==u)
            npix = len(ypix)
            stat.append(
                {'ypix': ypix,
                 'xpix': xpix,
                 'npix': npix,
                 'lam': np.ones(npix, np.float32),
                 'med': [np.mean(ypix), np.mean(xpix)]})
        stat = np.array(stat)
        stat = roi_stats(stat, ops['Ly'], ops['Lx'])
    return stat

