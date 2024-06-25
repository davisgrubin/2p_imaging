#!/usr/bin/env python3

import numpy as np
from skimage.segmentation import find_boundaries


# label images with yellow and green.
def get_labeled_masks_img(masks, labels, cate):
    labeled_masks_img = np.zeros(
        (masks.shape[0], masks.shape[1], 3), dtype='int32')
    neuron_idx = np.where(labels == cate)[0] + 1
    for i in neuron_idx:
        neuron_mask = ((masks == i) * 255).astype('int32')
        labeled_masks_img[:,:,0] += neuron_mask
    return labeled_masks_img


# get ROI sub image.
def get_roi_range(size, masks):
    row = []
    col = []
    for i in np.unique(masks)[1:]:
        rows, cols = np.where(masks == i)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        center_row = (min_row + max_row) // 2
        center_col = (min_col + max_col) // 2
        row.append(int(max(0, min(masks.shape[0] - size, center_row - size/2))))
        col.append(int(max(0, min(masks.shape[1] - size, center_col - size/2))))
    return row, col
         

# automatical adjustment of contrast.
def adjust_contrast(org_img, lower_percentile=50, upper_percentile=99):
    lower = np.percentile(org_img, lower_percentile)
    upper = np.percentile(org_img, upper_percentile)
    img = np.clip((org_img - lower) * 255 / (upper - lower), 0, 255)
    img = img.astype('int32')
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


class plotter_VIPTD_G8_masks:
    
    def __init__(
            self,
            labels,
            masks,
            mean_func, max_func,
            mean_anat
            ):
        self.labels = labels
        self.masks = masks
        self.mean_func = mean_func
        self.max_func = max_func
        self.mean_anat = mean_anat
        self.labeled_masks_img = get_labeled_masks_img(masks, labels, 1)
        self.unsure_masks_img = get_labeled_masks_img(masks, labels, 0)
        self.size = 128
        self.roi_row, self.roi_col = get_roi_range(self.size, masks)
        
    # functional channel max projection.
    def func_max(self, ax):
        func_img = np.zeros(
            (self.max_func.shape[0], self.max_func.shape[1], 3), dtype='int32')
        func_img[:,:,1] = adjust_contrast(self.max_func)
        func_img = adjust_contrast(func_img)
        x_all, y_all = np.where(find_boundaries(self.masks))
        for x,y in zip(x_all, y_all):
            func_img[x,y,:] = np.array([255,255,255])
        ax.matshow(func_img)
        adjust_layout(ax)
        ax.set_title('functional channel max projection image')

    # functional channel ROI masks.
    def func_masks(self, ax):
        masks_img = np.zeros(
            (self.masks.shape[0], self.masks.shape[1], 3), dtype='int32')
        masks_img[:,:,1] = self.masks
        masks_img[masks_img >= 1] = 255
        ax.matshow(masks_img)
        adjust_layout(ax)
        ax.set_title('functional channel ROI masks')

    # anatomy channel mean image.
    def anat_mean(self, ax):
        anat_img = np.zeros(
            (self.mean_anat.shape[0], self.mean_anat.shape[1], 3), dtype='int32')
        anat_img[:,:,0] = adjust_contrast(self.mean_anat)
        anat_img = adjust_contrast(anat_img)
        x_all, y_all = np.where(find_boundaries(self.masks))
        for x,y in zip(x_all, y_all):
            anat_img[x,y,:] = np.array([255,255,255])
        x_all, y_all = np.where(find_boundaries(self.labeled_masks_img[:,:,0]))
        for x,y in zip(x_all, y_all):
            anat_img[x,y,:] = np.array([255,255,0])
        x_all, y_all = np.where(find_boundaries(self.unsure_masks_img[:,:,0]))
        for x,y in zip(x_all, y_all):
            anat_img[x,y,:] = np.array([0,196,255])
        ax.matshow(anat_img)
        adjust_layout(ax)
        ax.set_title('anatomy channel mean image')

    # anatomy channel masks.
    def anat_label_masks(self, ax):
        ax.matshow(self.labeled_masks_img)
        adjust_layout(ax)
        ax.set_title('anatomy channel label masks')

    # superimpose image.
    def superimpose(self, ax):
        super_img = np.zeros(
            (self.max_func.shape[0], self.max_func.shape[1], 3), dtype='int32')
        super_img[:,:,0] = adjust_contrast(self.mean_anat)
        super_img[:,:,1] = adjust_contrast(self.max_func)
        super_img = adjust_contrast(super_img)
        x_all, y_all = np.where(find_boundaries(self.labeled_masks_img[:,:,0]))
        for x,y in zip(x_all, y_all):
            super_img[x,y,:] = np.array([255,255,255])
        ax.matshow(super_img)
        adjust_layout(ax)
        ax.set_title('channel images superimpose')

    # channel shared masks.
    def shared_masks(self, ax):
        label_masks = np.zeros(
            (self.masks.shape[0], self.masks.shape[1], 3), dtype='int32')
        label_masks[:,:,0] = get_labeled_masks_img(self.masks, self.labels, 1)[:,:,0]
        label_masks[:,:,1] = self.masks
        label_masks[:,:,2] = get_labeled_masks_img(self.masks, self.labels, 0)[:,:,0]
        label_masks[label_masks >= 1] = 255
        label_masks = label_masks.astype('int32')
        ax.matshow(label_masks)
        adjust_layout(ax)
        ax.set_title('channel masks superimpose')

    # ROI global location.
    def roi_loc(self, ax, roi_id):
        super_img = np.zeros(
            (self.max_func.shape[0], self.max_func.shape[1], 3), dtype='int32')
        super_img[:,:,0] = adjust_contrast(self.mean_anat)
        super_img[:,:,1] = adjust_contrast(self.max_func)
        super_img = adjust_contrast(super_img)
        x_all, y_all = np.where(self.masks==(roi_id+1))
        c_x = np.mean(x_all).astype('int32')
        c_y = np.mean(y_all).astype('int32')
        super_img[c_x,:,:] = np.array([128,128,255])
        super_img[:,c_y,:] = np.array([128,128,255])
        for x,y in zip(x_all, y_all):
            super_img[x,y,:] = np.array([255,255,255])
        ax.matshow(super_img)
        adjust_layout(ax)
        if self.labels[roi_id]==-1:
            c = 'excitory'
        if self.labels[roi_id]==0:
            c = 'unsure'
        if self.labels[roi_id]==1:
            c = 'inhibitory'
        ax.set_title('ROI # {} location ({})'.format(str(roi_id).zfill(4), c))
    
    # ROI functional channel max projection.
    def roi_func(self, ax, roi_id):
        r = self.roi_row[roi_id]
        c = self.roi_col[roi_id]
        max_func_img = self.max_func[r:r+self.size, c:c+self.size]
        roi_masks = (self.masks[r:r+self.size, c:c+self.size]==(roi_id+1))*1
        img = np.zeros((max_func_img.shape[0], max_func_img.shape[1], 3))
        img[:,:,1] = max_func_img
        img = adjust_contrast(img)
        x_all, y_all = np.where(find_boundaries(roi_masks))
        for x,y in zip(x_all, y_all):
            img[x,y,:] = np.array([255,255,255])
        ax.matshow(img)
        adjust_layout(ax)
        ax.set_title('functional channel max projection')
    
    # ROI anatomical channel mean projection.
    def roi_anat(self, ax, roi_id):
        r = self.roi_row[roi_id]
        c = self.roi_col[roi_id]
        mean_anat_img = self.mean_anat[r:r+self.size, c:c+self.size]
        roi_masks = (self.masks[r:r+self.size, c:c+self.size]==(roi_id+1))*1
        img = np.zeros((mean_anat_img.shape[0], mean_anat_img.shape[1], 3))
        img[:,:,0] = mean_anat_img
        img = adjust_contrast(img)
        x_all, y_all = np.where(find_boundaries(roi_masks))
        for x,y in zip(x_all, y_all):
            img[x,y,:] = np.array([255,255,255])
        ax.matshow(img)
        adjust_layout(ax)
        ax.set_title('anatomy channel mean image')

    # ROI channel image superimpose.
    def roi_superimpose(self, ax, roi_id):
        super_img = np.zeros(
            (self.max_func.shape[0], self.max_func.shape[1], 3), dtype='int32')
        super_img[:,:,0] = adjust_contrast(self.mean_anat)
        super_img[:,:,1] = adjust_contrast(self.max_func)
        super_img = adjust_contrast(super_img)
        r = self.roi_row[roi_id]
        c = self.roi_col[roi_id]
        super_img = super_img[r:r+self.size, c:c+self.size, :]
        roi_masks = (self.masks[r:r+self.size, c:c+self.size]==(roi_id+1))*1
        x_all, y_all = np.where(find_boundaries(roi_masks))
        for x,y in zip(x_all, y_all):
            super_img[x,y,:] = np.array([255,255,255])
        ax.matshow(super_img)
        adjust_layout(ax)
        ax.set_title('channel images superimpose')
    
    # ROI masks.
    def roi_masks(self, ax, roi_id):
        r = self.roi_row[roi_id]
        c = self.roi_col[roi_id]
        roi_masks = (self.masks[r:r+self.size, c:c+self.size]==(roi_id+1))*1
        img = np.zeros((roi_masks.shape[0], roi_masks.shape[1], 3), dtype='int32')
        x_all, y_all = np.where(roi_masks)
        for x,y in zip(x_all, y_all):
            img[x,y,:] = np.array([255,255,255])
        ax.matshow(img)
        adjust_layout(ax)
        ax.set_title('ROI masks')

