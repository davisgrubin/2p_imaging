# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:40:23 2024

@author: saminnaji3
"""

import os
import fitz
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

import warnings
warnings.filterwarnings("ignore")

from modules import Trialization
from modules import Alignment
from modules.ReadResults import read_masks
from modules.ReadResults import read_raw_voltages
#from modules.ReadResults import read_dff
from modules.ReadResults import read_neural_trials
from modules.ReadResults import read_move_offset
from modules.ReadResults import read_significance
def read_dff(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'dff.h5'), 'r')
    dff = np.array(f['name'])
    f.close()
    return dff

import h5py
# with h5py.File('C:/Users/saminnaji3/Downloads/2p imaging/Data/E4L7/20241030_crbl/dff.h5', 'r') as f:
    
#     dataset = f['name']  # Replace 'dataset_name' with the actual name
#     data = dataset[:]  # Read the data from the dataset
#     print(data)



############## reading neural data
cate_delay = 25
session_data_path = 'C:/Users/saminnaji3/Downloads/2p imaging/Data/E4L7/20241030_crbl'
ops = np.load(
    os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
    allow_pickle=True).item()

ops['save_path0'] = os.path.join(session_data_path)


# %%

print('===============================================')
print('=============== trial alignment ===============')
print('===============================================')
print('Reading dff traces and voltage recordings')
dff = read_dff(ops)
[vol_time, 
 vol_start, 
 vol_stim_vis, 
 vol_img, 
 vol_hifi, 
 vol_stim_aud, 
 vol_flir,
 vol_pmt, 
 vol_led] = read_raw_voltages(ops)
print('Correcting 2p camera trigger time')
# signal trigger time stamps.
time_img, _   = Trialization.get_trigger_time(vol_time, vol_img)
# correct imaging timing.
time_neuro = Trialization.correct_time_img_center(time_img)
# stimulus alignment.
print('Aligning stimulus to 2p frame')
stim = Trialization.align_stim(vol_time, time_neuro, vol_stim_vis, vol_stim_vis)
# trial segmentation.
print('Segmenting trials')
start, end = Trialization.get_trial_start_end(vol_time, vol_start)
neural_trials = Trialization.trial_split(
    start, end,
    dff, stim, time_neuro,
    vol_stim_vis, vol_time)
neural_trials = Trialization.trial_label(ops, neural_trials)
# save the final data.
print('Saving trial data')
Trialization.save_trials(ops, neural_trials)


# %%
def get_mean_sem(data):
    m = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    count = np.sum(~np.isnan(data.reshape(-1, data.shape[-1])), axis=0)
    s = std / np.sqrt(count)
    return m, s
l_frames = 5
r_frames = 50
state = 'LED'
[neu_seq, neu_time, stim_seq] =  Alignment.get_stim_response(
        neural_trials, state, 1,
        l_frames, r_frames)
[neu_mean,neu_sd] =  get_mean_sem(neu_seq)

l_frames = 5
r_frames = 50
state = 'AirPuff'
[neu_seq_1, neu_time1, stim_seq_1] =  Alignment.get_stim_response(
        neural_trials, state, 1,
        l_frames, r_frames)

[neu_mean1,neu_sd1] =  get_mean_sem(neu_seq_1)

l_frames = 2
r_frames = 100
state = 'ITI'
[neu_seq_2, neu_time2, stim_seq_2] =  Alignment.get_stim_response(
        neural_trials, state, 1,
        l_frames, r_frames)

[neu_mean2,neu_sd2] =  get_mean_sem(neu_seq_2)


l_frames = 5
r_frames = 50
state = 'LED'
[neu_seql ,  neu_timel, stim_seq_l] =  Alignment.get_stim_response(
        neural_trials, state, 2,
        l_frames, r_frames)
[neu_meanl,neu_sdl] =  get_mean_sem(neu_seql)

l_frames = 5
r_frames = 50
state = 'AirPuff'
[neu_seq_1l, neu_time1l, stim_seq_1l] =  Alignment.get_stim_response(
        neural_trials, state, 2,
        l_frames, r_frames)

[neu_mean1l,neu_sd1l]=  get_mean_sem(neu_seq_1l)

l_frames = 2
r_frames = 100
state = 'ITI'
[neu_seq_2l, neu_time2l, stim_seq_2l] =  Alignment.get_stim_response(
        neural_trials, state, 2,
        l_frames, r_frames)

[neu_mean2l,neu_sd2l]=  get_mean_sem(neu_seq_2l)
# %%

[labels, masks, mean_func, max_func, mean_anat, masks_anat, masks_anat_corrected] = read_masks(ops)

# %% plotting the mask and good ROIs  
from plot.fig1_mask import plotter_all_masks

[labels, masks, mean_func, max_func, mean_anat, masks_anat, masks_anat_corrected] = read_masks(ops)
plotter_masks = plotter_all_masks(
        labels, masks, mean_func, max_func, mean_anat, masks_anat)
fig = plt.figure(figsize=(105, 210))
gs = GridSpec(30, 15, figure=fig)
mask_ax01 = plt.subplot(gs[0:2, 0:2])
mask_ax02 = plt.subplot(gs[0:2, 2:4])
plotter_masks.func(mask_ax01, 'max')
plotter_masks.func_masks(mask_ax02)


# %%
from plot.utils import adjust_layout_neu

sig_tag = 'sig'
session_data_name = '20241030'
from plot.fig1_mask import plotter_all_masks
plotter_masks = plotter_all_masks(
        labels, masks, mean_func, max_func, mean_anat, masks_anat)

roi_report = fitz.open()
for roi_id in tqdm(np.argsort(labels, kind='stable')):
    
    fig = plt.figure(figsize=(70, 56))
    gs = GridSpec(8, 10, figure=fig)
    # masks.
    mask_ax01 = plt.subplot(gs[0:2, 0:2])
    mask_ax02 = plt.subplot(gs[0, 2])
    mask_ax03 = plt.subplot(gs[1, 2])

    plotter_masks.roi_loc_1chan(mask_ax01, roi_id, 'max')
    plotter_masks.roi_func(mask_ax02, roi_id, 'max')
    plotter_masks.roi_masks(mask_ax03, roi_id)
    # alignment.
    mask_ax01 = plt.subplot(gs[2, 0:2])
    mask_ax02 = plt.subplot(gs[2, 2:4])
    mask_ax03 = plt.subplot(gs[2, 4:6])
    mask_ax01.plot(neu_time , neu_mean[roi_id , :] ,color = 'dodgerblue')
    mask_ax01.fill_between(neu_time ,neu_mean[roi_id , :]-neu_sd[roi_id , :] ,neu_mean[roi_id , :]+neu_sd[roi_id , :]  ,color = 'dodgerblue' , alpha = 0.2)
    mask_ax01.fill_betweenx([np.min(neu_mean[roi_id , :]-neu_sd[roi_id , :]) , np.max(neu_mean[roi_id , :]+neu_sd[roi_id , :])] , 0 , stim_seq , color = 'gray', alpha = 0.2)
    mask_ax01.set_title('LED')
    adjust_layout_neu(mask_ax01)
    mask_ax01.fill_betweenx([np.min(neu_mean[roi_id , :]-neu_sd[roi_id , :]) , np.max(neu_mean[roi_id , :]+neu_sd[roi_id , :])] , 200,200 + stim_seq_1 , color = 'gray', alpha = 0.2)
    adjust_layout_neu(mask_ax02)
    mask_ax02.plot(neu_time2, neu_mean2[roi_id , :] ,color =  'dodgerblue')
    mask_ax02.fill_between(neu_time2 ,neu_mean2[roi_id , :]-neu_sd2[roi_id , :] ,neu_mean2[roi_id , :]+neu_sd2[roi_id , :]  ,color = 'dodgerblue' , alpha = 0.2)
    mask_ax02.fill_betweenx([np.min(neu_mean2[roi_id , :]-neu_sd2[roi_id , :]) , np.max(neu_mean2[roi_id , :]+neu_sd2[roi_id , :])] , 0 , stim_seq_2 , color = 'gray', alpha = 0.2)
    mask_ax02.set_title('ITI')
    adjust_layout_neu(mask_ax02)
    
    mask_ax01 = plt.subplot(gs[3, 0:2])
    mask_ax02 = plt.subplot(gs[3, 2:4])
    mask_ax03 = plt.subplot(gs[3, 4:6])
    mask_ax01.plot(neu_time , neu_meanl[roi_id , :] ,color = 'dodgerblue')
    mask_ax01.fill_between(neu_time ,neu_meanl[roi_id , :]-neu_sdl[roi_id , :] ,neu_meanl[roi_id , :]+neu_sdl[roi_id , :]  ,color = 'dodgerblue' , alpha = 0.2)
    mask_ax01.fill_betweenx([np.min(neu_meanl[roi_id , :]-neu_sdl[roi_id , :]) , np.max(neu_meanl[roi_id , :]+neu_sdl[roi_id , :])] , 0 , stim_seq_l , color = 'gray', alpha = 0.2)
    mask_ax01.set_title('LED , long')
    adjust_layout_neu(mask_ax01)
    mask_ax01.fill_betweenx([np.min(neu_meanl[roi_id , :]-neu_sdl[roi_id , :]) , np.max(neu_meanl[roi_id , :]+neu_sdl[roi_id , :])] , 400,400 + stim_seq_1l , color = 'gray', alpha = 0.2)
    adjust_layout_neu(mask_ax02)
    mask_ax02.plot(neu_time2l, neu_mean2l[roi_id , :] ,color =  'dodgerblue')
    mask_ax02.fill_between(neu_time2l ,neu_mean2l[roi_id , :]-neu_sd2l[roi_id , :] ,neu_mean2l[roi_id , :]+neu_sd2l[roi_id , :]  ,color = 'dodgerblue' , alpha = 0.2)
    mask_ax02.fill_betweenx([np.min(neu_mean2l[roi_id , :]-neu_sd2l[roi_id , :]) , np.max(neu_mean2l[roi_id , :]+neu_sd2l[roi_id , :])] , 0 , stim_seq_2l , color = 'gray', alpha = 0.2)
    mask_ax02.set_title('ITI , long')
    adjust_layout_neu(mask_ax02)
    # save figure.
    fname = os.path.join(
        ops['save_path0'], 'figures',
        str(roi_id).zfill(4)+'.pdf')
    fig.set_size_inches(70, 56)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    roi_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    
    
roi_report.save(os.path.join( ops['save_path0'], 'figures', 'roi_report_{}_{}.pdf'.format(sig_tag, session_data_name)))
roi_report.close()

# %%


from plot.fig1_mask import plotter_all_masks
plotter_masks = plotter_all_masks(
        labels, masks, mean_func, max_func, mean_anat, masks_anat)
fig = plt.figure(figsize=(70, 56))
gs = GridSpec(8, 10, figure=fig)
mask_ax01 = plt.subplot(gs[0:2, 0:2])
mask_ax02 = plt.subplot(gs[0:2, 2:4])
plotter_masks.func(mask_ax01, 'max')
plotter_masks.func_masks(mask_ax02)


good = [2,4,6,7,8,11,13,14,16,17,18,19,20,21,25,26,29,30,31,34,36,37,39,42,43,48,49,50]
both = [8,11,13,14,17,18,19,20,25,26,34,39,42,48,49,50]
LED =[7,16,18,21,25,29,30,31,36,37,43]
sb = []
lb = []
for roi_id in tqdm(LED):
    sb.append(neu_seq[:,roi_id,:])
    lb.append(neu_seql[:,roi_id,:])

sb = np.array(sb).mean(axis=0)
sb = sb.mean(axis=0)

lb = np.array(lb).mean(axis=0)
lb = lb.mean(axis=0)


roi_report = fitz.open()
plotter_masks = plotter_all_masks(
        labels, masks, mean_func, max_func, mean_anat, masks_anat)
fig = plt.figure(figsize=(105, 210))
gs = GridSpec(30, 15, figure=fig)
mask_ax01 = plt.subplot(gs[0:2, 0:2])
mask_ax02 = plt.subplot(gs[0:2, 2:4])
plotter_masks.func(mask_ax01, 'max')
plotter_masks.func_masks(mask_ax02)
mask_ax01 = plt.subplot(gs[2, 0:2])

mask_ax01.plot(neu_time ,sb,color = 'dodgerblue')
mask_ax01.fill_betweenx([np.min(sb) , np.max(sb)] , 0 , stim_seq , color = 'gray', alpha = 0.2)
mask_ax01.set_title('LED')

mask_ax01.fill_betweenx([np.min(sb) , np.max(sb)] , 200, stim_seq+200,color = 'gray', alpha = 0.2)



mask_ax01 = plt.subplot(gs[3, 0:2])

mask_ax01.plot(neu_timel,lb,color = 'dodgerblue')
mask_ax01.fill_betweenx([np.min(lb) , np.max(lb)] , 0 , stim_seq , color = 'gray', alpha = 0.2)
mask_ax01.set_title('LED,long')

mask_ax01.fill_betweenx([np.min(lb) , np.max(lb)] , 400, stim_seq+400,color = 'gray', alpha = 0.2)

 

fname = os.path.join(
ops['save_path0'], 'figures',
str(roi_id).zfill(4)+'.pdf')
fig.set_size_inches(70, 56)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
roi_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
   
roi_report.save(os.path.join( ops['save_path0'], 'figures', 'roi_rept_{}.pdf'.format( session_data_name)))
roi_report.close()