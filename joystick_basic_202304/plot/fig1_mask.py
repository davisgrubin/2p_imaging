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
    labeled_masks_img = np.zeros((masks.shape[0], masks.shape[1], 3), dtype='uint8')
    neuron_idx = np.where(labels == 1)[0] + 1
    for i in neuron_idx:
        neuron_mask = ((masks == i) * 255).astype('uint8')
        labeled_masks_img[:,:,0] += neuron_mask
    return labeled_masks_img


# functional channel max projection.

def plot_ppc_func_max(ax, max_func, masks):
    func_img = np.zeros((max_func.shape[0], max_func.shape[1], 3), dtype='uint8')
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
    anat_img = np.zeros((mean_anat.shape[0], mean_anat.shape[1], 3), dtype='uint8')
    anat_img[:,:,0] = adjust_contrast(mean_anat)
    anat_img = adjust_contrast(anat_img)
    x_all, y_all = np.where(find_boundaries(masks))
    for x,y in zip(x_all, y_all):
        anat_img[x,y,:] = np.array([255,255,255])
    labeled_masks_img = get_labeled_masks_img(masks, labels)
    x_all, y_all = np.where(find_boundaries(labeled_masks_img[:,:,0]))
    for x,y in zip(x_all, y_all):
        anat_img[x,y,:] = np.array([255,255,0])
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
    super_img = np.zeros((max_func.shape[0], max_func.shape[1], 3), dtype='uint8')
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
    

# ROI global location.

def plot_ppc_roi_loc(ax, roi_id, max_func, mean_anat, masks, labels):
    super_img = np.zeros((max_func.shape[0], max_func.shape[1], 3), dtype='uint8')
    super_img[:,:,0] = adjust_contrast(mean_anat)
    super_img[:,:,1] = adjust_contrast(max_func)
    super_img = adjust_contrast(super_img)
    x_all, y_all = np.where(masks==(roi_id+1))
    c_x = np.mean(x_all).astype('uint8')
    c_y = np.mean(y_all).astype('uint8')
    super_img[c_x,:,:] = np.array([0,0,255])
    super_img[:,c_y,:] = np.array([0,0,255])
    for x,y in zip(x_all, y_all):
        super_img[x,y,:] = np.array([255,255,255])
    ax.matshow(super_img)
    adjust_layout(ax)
    if labels[roi_id]==-1:
        c = 'excitory'
    if labels[roi_id]==0:
        c = 'unsure'
    if labels[roi_id]==1:
        c = 'inhibitory'
    ax.set_title('ROI # {} location ({})'.format(str(roi_id).zfill(4),c))
    

# ROI functional channel max projection.

def plot_ppc_roi_func(ax, roi_id, max_func, masks):
    size = 128
    rows, cols = np.where(masks == (roi_id+1))
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    r = int(max(0, min(masks.shape[0] - size, center_row - size/2)))
    c = int(max(0, min(masks.shape[1] - size, center_col - size/2)))
    max_func_img = max_func[r:r+size, c:c+size]
    roi_masks = (masks[r:r+size, c:c+size]==(roi_id+1))*1
    img = np.zeros((max_func_img.shape[0], max_func_img.shape[1], 3))
    img[:,:,1] = max_func_img
    img = adjust_contrast(img)
    x_all, y_all = np.where(find_boundaries(roi_masks))
    for x,y in zip(x_all, y_all):
        img[x,y,:] = np.array([255,255,255])
    ax.matshow(img)
    adjust_layout(ax)
    ax.set_title('ROI # {} functional channel max projection image'.format(
        str(roi_id).zfill(4)))
    

# ROI anatomical channel mean projection.

def plot_ppc_roi_anat(ax, roi_id, mean_anat, masks):
    size = 128
    rows, cols = np.where(masks == (roi_id+1))
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    r = int(max(0, min(masks.shape[0] - size, center_row - size/2)))
    c = int(max(0, min(masks.shape[1] - size, center_col - size/2)))
    mean_anat_img = mean_anat[r:r+size, c:c+size]
    roi_masks = (masks[r:r+size, c:c+size]==(roi_id+1))*1
    img = np.zeros((mean_anat_img.shape[0], mean_anat_img.shape[1], 3))
    img[:,:,0] = mean_anat_img
    img = adjust_contrast(img)
    x_all, y_all = np.where(find_boundaries(roi_masks))
    for x,y in zip(x_all, y_all):
        img[x,y,:] = np.array([255,255,255])
    ax.matshow(img)
    adjust_layout(ax)
    ax.set_title('ROI # {} anatomy channel mean image'.format(
        str(roi_id).zfill(4)))


# ROI channel image superimpose.

def plot_ppc_roi_superimpose(ax, roi_id, max_func, mean_anat, masks):
    size = 128
    super_img = np.zeros((max_func.shape[0], max_func.shape[1], 3), dtype='uint8')
    super_img[:,:,0] = adjust_contrast(mean_anat)
    super_img[:,:,1] = adjust_contrast(max_func)
    super_img = adjust_contrast(super_img)
    rows, cols = np.where(masks == (roi_id+1))
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    r = int(max(0, min(masks.shape[0] - size, center_row - size/2)))
    c = int(max(0, min(masks.shape[1] - size, center_col - size/2)))
    super_img = super_img[r:r+size, c:c+size, :]
    roi_masks = (masks[r:r+size, c:c+size]==(roi_id+1))*1
    x_all, y_all = np.where(find_boundaries(roi_masks))
    for x,y in zip(x_all, y_all):
        super_img[x,y,:] = np.array([255,255,255])
    ax.matshow(super_img)
    adjust_layout(ax)
    ax.set_title('ROI # {} channel images superimpose'.format(
        str(roi_id).zfill(4)))
    
    
# ROI masks.

def plot_ppc_roi_masks(ax, roi_id, masks):
    size = 128
    rows, cols = np.where(masks == (roi_id+1))
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    r = int(max(0, min(masks.shape[0] - size, center_row - size/2)))
    c = int(max(0, min(masks.shape[1] - size, center_col - size/2)))
    roi_masks = (masks[r:r+size, c:c+size]==(roi_id+1))*1
    img = np.zeros((roi_masks.shape[0], roi_masks.shape[1], 3), dtype='uint8')
    x_all, y_all = np.where(roi_masks)
    for x,y in zip(x_all, y_all):
        img[x,y,:] = np.array([255,255,255])
    ax.matshow(img)
    adjust_layout(ax)
    ax.set_title('ROI # {} masks'.format(
        str(roi_id).zfill(4)))

    