#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

from postprocess.ReadResults import read_move_offset


# main function for plot.

def plot_san1(ops):

    try:
        print('plotting sanity check 1 motion offsets')

        # read offsets.
        xoff, yoff = read_move_offset(ops)

        fig, axs = plt.subplots(2, 1, figsize=(4, 8))

        # read mask from in save_path0 in ops.
        mask = read_mask(ops)
        func_ch    = mask['ch'+str(ops['functional_chan'])]['input_img']
        func_masks = mask['ch'+str(ops['functional_chan'])]['masks']
        anat_ch    = mask['ch'+str(3-ops['functional_chan'])]['input_img']
        anat_masks = mask['ch'+str(3-ops['functional_chan'])]['masks']


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
