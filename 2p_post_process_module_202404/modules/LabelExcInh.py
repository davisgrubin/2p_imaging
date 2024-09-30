#!/usr/bin/env python3

import os
import h5py
import tifffile
import numpy as np
from tqdm import tqdm
from cellpose import models
from cellpose import io
import matplotlib.pyplot as plt
from scipy.stats import linregress
from skimage.segmentation import find_boundaries

# from .visualization_VIPTD_G8 import plot_js_VIPTD_G8

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
    masks_func = masks_npy[y1:y2, x1:x2]
    mean_func = ops['meanImg'][y1:y2, x1:x2]
    max_func = ops['max_proj']
    if ops['nchannels'] == 2:
        mean_anat = ops['meanImg_chan2'][y1:y2, x1:x2]
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
    masks_3d = np.zeros(
        (len(anat_roi_ids), masks_anat.shape[0], masks_anat.shape[1]))
    for i, roi_id in enumerate(anat_roi_ids):
        masks_3d[i] = (masks_anat == roi_id).astype(int)
    masks_3d[masks_3d != 0] = 1
    # compute relative overlaps coefficient for each functional roi.
    prob = []
    for i in tqdm(np.unique(masks_func)[1:]):
        roi_masks = (masks_func == i).astype('int32')
        roi_masks_tile = np.tile(
            np.expand_dims(roi_masks, 0),
            (len(anat_roi_ids), 1, 1))
        overlap = (roi_masks_tile * masks_3d).reshape(len(anat_roi_ids), -1)
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

def save_masks(ops, masks_func, masks_anat, masks_anat_corrected, mean_func, max_func, mean_anat, labels):
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
        f['masks_anat_corrected'] = masks_anat_corrected
    f.close()


# function to remove green from red channel
# uses linear regression to fit the green channel to the red channel

def compute_offset_slope(mean_anat, mean_func, ops):
    # check if slope and offset already computed
    if os.path.exists(os.path.join(ops['save_path0'], 'slope_offset.npy')):
        slope, offset = np.load(os.path.join(
            ops['save_path0'], 'slope_offset.npy'))
    else:
        # compute slope and offset
        # fit func = slope * anat + offset
        print("mean_anat.shape: ", mean_anat.shape)
        print("mean_func.shape: ", mean_func.shape)
        # flatten the arrays
        mean_anat_flat = mean_anat.flatten()
        mean_func_flat = mean_func.flatten()
        slope, offset, r_value, p_value, std_err = linregress(
            mean_func_flat, mean_anat_flat)
        # save slope and offset
        f = h5py.File(
            os.path.join(ops['save_path0'], 'masks.h5'),
            'w')
        f['slope'] = slope
        f['offset'] = offset
        f.close()
    return slope, offset, r_value, p_value, std_err


def remove_green_bleedthrough(offset, slope, mean_func, mean_anat):
    # corrected functional channel = original functional channel - slope * original anatomical channel
    mean_anat_new = mean_anat - (slope * mean_func)
    return mean_anat_new

# main function to use anatomical to label functional channel masks.


# def create_rgb_image(image, color):
#     rgb = np.zeros((*image.shape, 3))
#     for i, c in enumerate(color):
#         rgb[:, :, i] = image * c / np.max(image)
#     return rgb

# adjust layout for masks plot.
def adjust_layout(ax):
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


# label images with yellow and green.
def get_labeled_masks_img(masks, labels, cate):
    labeled_masks_img = np.zeros(
        (masks.shape[0], masks.shape[1], 3), dtype='int32')
    neuron_idx = np.where(labels == cate)[0] + 1
    for i in neuron_idx:
        neuron_mask = ((masks == i) * 255).astype('int32')
        labeled_masks_img[:, :, 0] += neuron_mask
    return labeled_masks_img

# automatical adjustment of contrast.


def adjust_contrast(org_img, lower_percentile=50, upper_percentile=99):
    lower = np.percentile(org_img, lower_percentile)
    upper = np.percentile(org_img, upper_percentile)
    img = np.clip((org_img - lower) * 255 / (upper - lower), 0, 255)
    img = img.astype('int32')
    return img


def anat(ax, mean_anat, masks, labeled_masks_img, unsure_masks_img, with_mask=True, title='anatomy channel mean image'):
    anat_img = np.zeros(
        (mean_anat.shape[0], mean_anat.shape[1], 3), dtype='int32')
    anat_img[:, :, 0] = adjust_contrast(mean_anat)
    anat_img = adjust_contrast(anat_img)
    if with_mask:
        x_all, y_all = np.where(find_boundaries(masks))
        for x, y in zip(x_all, y_all):
            anat_img[x, y, :] = np.array([255, 255, 255])
        x_all, y_all = np.where(find_boundaries(
            labeled_masks_img[:, :, 0]))
        for x, y in zip(x_all, y_all):
            anat_img[x, y, :] = np.array([255, 255, 0])
        x_all, y_all = np.where(find_boundaries(
            unsure_masks_img[:, :, 0]))
        for x, y in zip(x_all, y_all):
            anat_img[x, y, :] = np.array([0, 196, 255])
    ax.matshow(anat_img)
    adjust_layout(ax)
    ax.set_title(f'{title}')


def run(ops, diameter):
    print('===============================================')
    print('===== two channel data roi identification =====')
    print('===============================================')
    print('Reading masks in functional channel')
    [masks_func, mean_func, max_func, mean_anat] = get_mask(ops)
    # deal with data here
    if np.max(masks_func) == 0:
        raise ValueError('No masks found.')
    if ops['nchannels'] == 1:
        print('Single channel recording so skip ROI labeling')
        labels = -1 * np.ones(int(np.max(masks_func))).astype('int32')
        save_masks(ops, masks_func, None, mean_func, max_func, None, labels)
    else:
        # remove green from red channel here
        print('Running cellpose on anatomical channel mean image')
        print('Found diameter as {}'.format(diameter))

        # function to remove green from red channel
        slope, offset, _, _, _ = compute_offset_slope(
            mean_anat, mean_func, ops)

        mean_anat_corrected = remove_green_bleedthrough(
            offset, slope, mean_func, mean_anat)

        masks_anat = run_cellpose(ops, mean_anat, diameter)
        masks_anat_corrected = run_cellpose(ops, mean_anat_corrected, diameter)

        # visualize the corrected image against original green and red channel
        # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # ax[0].imshow(create_rgb_image(mean_anat, [0, 1, 0]))
        # ax[0].set_title('Original Anatomical Channel')
        # ax[1].imshow(create_rgb_image(mean_func, [1, 0, 0]))
        # ax[1].set_title('Original Functional Channel')
        # ax[2].imshow(create_rgb_image(mean_func_corrected, [1, 0, 0]))
        # ax[2].set_title('Corrected Functional Channel')
        # plt.show()

        print('Computing labels for each ROI')
        labels = get_label(masks_func, masks_anat)

        print('Computing corrected labels for each ROI')
        labels_corrected = get_label(masks_func, masks_anat_corrected)

        labeled_masks_img_orig = get_labeled_masks_img(masks_anat, labels, 1)
        labeled_masks_img_corr = get_labeled_masks_img(
            masks_anat_corrected, labels_corrected, 1)

        unsure_masks_img_orig = get_labeled_masks_img(masks_anat, labels, 0)
        unsure_masks_img_corr = get_labeled_masks_img(
            masks_anat_corrected, labels_corrected, 0)

        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        # pad between subplots
        plt.subplots_adjust(wspace=0.1)

        anat(ax[0], mean_anat, masks_anat, labeled_masks_img_orig,
             unsure_masks_img_orig, with_mask=True, title='Original + Mask')

        anat(ax[1], mean_anat_corrected, masks_anat_corrected, labeled_masks_img_corr,
             unsure_masks_img_corr, with_mask=True, title='Corrected + Masks')

        anat(ax[2], mean_anat, masks_anat, labeled_masks_img_orig,
             unsure_masks_img_orig, with_mask=False, title='Original')

        anat(ax[3], mean_anat_corrected, masks_anat_corrected, labeled_masks_img_corr,
             unsure_masks_img_corr, with_mask=False, title='Corrected')

        plt.rcParams['savefig.dpi'] = 1000
        # plot name with session data path
        plt.savefig(
            f'bleedthrough_channel_comparison_FN16_P_20240626_js_t.pdf')
        # plt.show()

        mean_anat = mean_anat_corrected
        masks_anat = masks_anat_corrected

        print('Found {} labeled ROIs out of {} in total'.format(
            np.sum(labels == 1), len(labels)))

        save_masks(
            ops,
            masks_func, masks_anat, masks_anat_corrected, mean_func,
            max_func, mean_anat,
            labels)

        print('Masks results saved')
