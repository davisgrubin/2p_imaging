#!/usr/bin/env python3

import numpy as np
from skimage.segmentation import find_boundaries


#%% utils

# automatical adjustment of contrast.

def adjust_contrast(org_img, lower_percentile=50, upper_percentile=99):
    lower = np.percentile(org_img, lower_percentile)
    upper = np.percentile(org_img, upper_percentile)
    img = np.clip((org_img - lower) * 255 / (upper - lower), 0, 255)
    img = img.astype(np.uint8)
    return img


# adjust layout for masks plot.

def adjust_layout(ax):
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


#%% ppc

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


# functional channel max projection.

def plot_ppc_func_max(ax, max_func, masks):
    func_img = np.zeros((max_func.shape[0], max_func.shape[1], 3))
    func_img[:,:,1] = adjust_contrast(max_func)
    func_img = adjust_contrast(func_img)
    x_all, y_all = np.where(find_boundaries(masks))
    for x,y in zip(x_all, y_all):
        func_img[x,y,:] = np.array([255,255,255])
    ax.matshow(func_img)
    adjust_layout(ax)
    ax.set_title('functional channel max projection image')


# functional channel ROI masks.

def plot_ppc_func_masks(ax, masks):
    masks_img = np.zeros((masks.shape[0], masks.shape[1], 3))
    masks_img[:,:,1] = masks
    masks_img[masks_img >= 1] = 255
    ax.matshow(masks_img)
    adjust_layout(ax)
    ax.set_title('functional channel ROI masks')


# anatomy channel mean image.

def plot_ppc_anat_mean(ax, mean_anat, masks, labels):
    anat_img = np.zeros((mean_anat.shape[0], mean_anat.shape[1], 3))
    anat_img[:,:,0] = adjust_contrast(mean_anat)
    anat_img = adjust_contrast(anat_img)
    labeled_masks_img = get_labeled_masks_img(masks, labels)
    x_all, y_all = np.where(find_boundaries(labeled_masks_img[:,:,0]))
    for x,y in zip(x_all, y_all):
        anat_img[x,y,:] = np.array([255,255,255])
    ax.matshow(anat_img)
    adjust_layout(ax)
    ax.set_title('anatomy channel mean image')


# anatomy channel masks.

def plot_ppc_anat_label_masks(ax, masks, labels):
    labeled_masks_img = get_labeled_masks_img(masks, labels)
    ax.matshow(labeled_masks_img)
    adjust_layout(ax)
    ax.set_title('anatomy channel label masks')


# superimpose image.

def plot_ppc_superimpose(ax, max_func, mean_anat, masks, labels):
    super_img = np.zeros((max_func.shape[0], max_func.shape[1], 3))
    super_img[:,:,0] = adjust_contrast(mean_anat)
    super_img[:,:,1] = adjust_contrast(max_func)
    super_img = adjust_contrast(super_img)
    labeled_masks_img = get_labeled_masks_img(masks, labels)
    x_all, y_all = np.where(find_boundaries(labeled_masks_img[:,:,0]))
    for x,y in zip(x_all, y_all):
        super_img[x,y,:] = np.array([255,255,255])
    ax.matshow(super_img)
    adjust_layout(ax)
    ax.set_title('channel images superimpose')


# channel shared masks.

def plot_ppc_shared_masks(ax, masks, labels):
    label_masks = np.zeros((masks.shape[0], masks.shape[1], 3))
    label_masks[:,:,0] = get_labeled_masks_img(masks, labels)[:,:,0]
    label_masks[:,:,1] = masks
    label_masks[label_masks >= 1] = 255
    label_masks = label_masks.astype('uint8')
    ax.matshow(label_masks)
    adjust_layout(ax)
    ax.set_title('channel masks superimpose')


#%% crbl

'''
# main function for plot.

def plot_fig1(ops):

    try:
        print('plotting fig1 masks')

        # read mask from in save_path0 in ops.
        [labels,
         masks,
         mean_func, max_func,
         mean_anat] = read_masks(ops)

        # 1 channel data.
        if ops['nchannels'] == 1:

            # plot figs.
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            plt.subplots_adjust(hspace=0.6)
            plt.subplots_adjust(wspace=0.6)

            # mean image.
            axs[0].matshow(
                adjust_contrast(mean_func),
                cmap=LinearSegmentedColormap.from_list(
                    "black_green", [(0, 0, 0), (0, 1, 0)]))
            axs[0].set_title('mean')

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

        # save figure
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures',
            'fig1_mask.pdf'),
            dpi=300)
        plt.close()

    except:
        print('plotting fig1 failed')
'''
