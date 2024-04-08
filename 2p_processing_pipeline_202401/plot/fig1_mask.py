#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

from postprocess.ReadResults import read_masks


# automatical adjustment of contrast.

def adjust_contrast(org_img, lower_percentile=50, upper_percentile=99):
    lower = np.percentile(org_img, lower_percentile)
    upper = np.percentile(org_img, upper_percentile)
    img = np.clip((org_img - lower) * 255 / (upper - lower), 0, 255)
    img = img.astype(np.uint8)
    return img


# overlap masks from two channels.

def overlap_label_masks(
        masks,
        labels
        ):
    img = np.zeros((masks.shape[0], masks.shape[1], 3))
    img[:,:,0] = get_labeled_masks_img(masks, labels)[:,:,0]
    img[:,:,1] = masks
    img[img >= 1] = 255
    img = img.astype('uint8')
    return img


# label images with yellow and green.

def get_labeled_masks_img(
        masks,
        labels,
        ):
    labeled_masks_img = np.zeros((masks.shape[0], masks.shape[1], 3))
    neuron_idx = np.where(labels == 1)[0] + 1
    for i in neuron_idx:
        neuron_mask = ((masks == i) * 255).astype('uint8')
        labeled_masks_img[:,:,0] += neuron_mask
    return labeled_masks_img


# superimpose functional and anatoical channel images.

def overlap_func_anat(mean_func, mean_anat):
    img = np.zeros((mean_func.shape[0], mean_func.shape[1], 3))
    img[:,:,0] = adjust_contrast(mean_anat)
    img[:,:,1] = adjust_contrast(mean_func)
    img = adjust_contrast(img)
    return img


# main function for plot.

def plot_fig1(ops):

    try:
        print('plotting fig1 masks')

        # read mask from in save_path0 in ops.
        [labels,
         masks,
         mean_func, max_func,
         ref_img,
         mean_anat] = read_masks(ops)

        # 1 channel data.
        if ops['nchannels'] == 1:

            # plot figs.
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            plt.subplots_adjust(hspace=0.6)
            plt.subplots_adjust(wspace=0.6)

            # reference image same as mean.
            axs[0].matshow(
                adjust_contrast(ref_img),
                cmap=LinearSegmentedColormap.from_list(
                    "black_green", [(0, 0, 0), (0, 1, 0)]))
            axs[0].set_title('reference image')

            # max projection.
            axs[1].matshow(
                adjust_contrast(max_func),
                cmap=LinearSegmentedColormap.from_list(
                    "black_green", [(0, 0, 0), (0, 1, 0)]))
            axs[1].set_title('max projection')

            # ROI masks.
            colors = plt.cm.nipy_spectral(np.linspace(0, 1, int(np.max(masks)+1)))
            np.random.shuffle(colors)
            colors[0,:] = [0,0,0,1]
            cmap = ListedColormap(colors)
            axs[2].matshow(masks, cmap=cmap)
            axs[2].set_title('ROI masks')

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
            axs[0,0].matshow(
                adjust_contrast(max_func),
                cmap=LinearSegmentedColormap.from_list(
                    "black_green", [(0, 0, 0), (0, 1, 0)]))
            axs[0,0].set_title('functional channel max projection image')

            # anatomy channel mean image.
            axs[0,1].matshow(
                adjust_contrast(mean_anat),
                cmap=LinearSegmentedColormap.from_list(
                    "black_red", [(0, 0, 0), (1, 0, 0)]))
            axs[0,1].set_title('anatomy channel mean image')

            # superimpose image.
            axs[0,2].matshow(overlap_func_anat(max_func, mean_anat))
            axs[0,2].set_title('channel images superimpose')

            # functional channel ROI masks.
            colors = plt.cm.nipy_spectral(np.linspace(0, 1, int(np.max(masks)+1)))
            np.random.shuffle(colors)
            colors[0,:] = [0,0,0,1]
            cmap = ListedColormap(colors)
            axs[1,0].matshow(masks, cmap=cmap)
            axs[1,0].set_title('functional channel ROI masks')

            # anatomy channel masks.
            axs[1,1].matshow(get_labeled_masks_img(masks, labels))
            axs[1,1].set_title('anatomy channel label masks')

            # channel shared masks.
            axs[1,2].imshow(overlap_label_masks(masks, labels))
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
