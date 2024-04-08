#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

from postprocess.ReadData import read_masks


# automatical adjustment of contrast.

def adjust_contrast(org_img, lower_percentile=75, upper_percentile=99):
    lower = np.percentile(org_img, lower_percentile)
    upper = np.percentile(org_img, upper_percentile)
    img = np.clip((org_img - lower) * 255 / (upper - lower), 0, 255)
    img = img.astype(np.uint8)
    return img


# convert matrix to RGB image.

def matrix_to_img(
        matrix,
        channel,
        binary
        ):
    img = np.zeros((matrix.shape[0], matrix.shape[1], 3))
    for ch in channel:
        img[:,:,ch] = matrix
    if binary:
        img[img >= 1] = 255
    else:
        img = adjust_contrast(img)
    img = img.astype('uint8')
    return img


# label images with yellow and green.

def get_labeled_masks_img(
        func_masks,
        label,
        ):
    # get green image.
    labeled_masks_img = matrix_to_img(func_masks, [1], True)
    # find marked neurons.
    neuron_idx = np.where(label == 1)[0] + 1
    for i in neuron_idx:
        neuron_mask = ((func_masks == i) * 255).astype('uint8')
        labeled_masks_img[:,:,0] += neuron_mask
    return labeled_masks_img


# main function for plot.

def plot_fig1(ops):

    try:
        print('plotting fig1 masks')
        
        # read mask from in save_path0 in ops.
        masks = read_masks(ops)
        func_ch    = masks['ch'+str(ops['functional_chan'])]['ref_img']
        func_masks = masks['masks']
        anat_ch    = masks['ch'+str(3-ops['functional_chan'])]['ref_img']
        anat_masks = masks['masks']

        # functional channel in green.
        func_ch_img = matrix_to_img(func_ch, [1], False)
        # anatomy channel in red.
        anat_ch_img = matrix_to_img(anat_ch, [0], False)
        # superimpose channel.
        super_img = func_ch_img + anat_ch_img
    
        # functional masks in green.
        func_masks_img = matrix_to_img(func_masks, [1], True)
        # anatomy masks in red.
        anat_masks_img = matrix_to_img(anat_masks, [0], True)
        # labelled masks.
        #super_masks = func_masks_img + anat_masks_img
        #super_masks = get_labeled_masks_img(func_masks, label)
        
        # 1 channel data.
        if ops['nchannels'] == 1:
    
            # plot figs.
            fig, axs = plt.subplots(2, 1, figsize=(4, 8))
    
            # functional channel mean image.
            axs[0].imshow(func_ch_img)
            axs[0].set_title('functional channel')
    
            # functional channel masks.
            axs[1].imshow(func_masks_img)
            axs[1].set_title('functional channel masks')
            
            # adjust layout
            for i in range(axs.shape[0]):
                axs[i].tick_params(tick1On=False)
                axs[i].spines['left'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['bottom'].set_visible(False)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
            fig.suptitle('Channel images and masks by suite2p')
            fig.tight_layout()
            
        # 2 channel data.
        if ops['nchannels'] == 2:
    
            # plot figs.
            fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    
            # functional channel mean image.
            axs[0,0].imshow(func_ch_img)
            axs[0,0].set_title('functional channel')
            # anatomy channel mean image.
            axs[0,1].imshow(anat_ch_img)
            axs[0,1].set_title('anatomy channel')
            # superimpose image.
            axs[0,2].imshow(super_img)
            axs[0,2].set_title('channel images superimpose')
    
            # functional channel masks.
            axs[1,0].imshow(func_masks_img)
            axs[1,0].set_title('functional channel masks')
            # anatomy channel masks.
            axs[1,1].imshow(anat_masks_img)
            axs[1,1].set_title('anatomy channel masks')
            # channel shared masks.
            #axs[1,2].imshow(super_masks)
            axs[1,2].set_title('channel masks superimpose')
    
            # adjust layout
            for i in range(axs.shape[0]):
                for j in range(axs.shape[1]):
                    axs[i,j].tick_params(tick1On=False)
                    axs[i,j].spines['left'].set_visible(False)
                    axs[i,j].spines['right'].set_visible(False)
                    axs[i,j].spines['top'].set_visible(False)
                    axs[i,j].spines['bottom'].set_visible(False)
                    axs[i,j].set_xticks([])
                    axs[i,j].set_yticks([])
            fig.suptitle('Channel images and masks by cellpose')
            fig.tight_layout()
    
        # save figure
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures',
            'fig1_mask.pdf'),
            dpi=300)
        plt.close()
        
    except:
        print('plotting fig1 failed')