# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:18:08 2025

@author: saminnaji3
"""

import os
import fitz
import csv
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

import warnings
warnings.filterwarnings("ignore")
from plot import coactivation_fraction
from modules import Trialization_Opto
from modules import FlirFrames
from modules import Alignment
from modules.ReadResults import read_masks
from modules.ReadResults import read_raw_voltages
from modules.ReadResults import read_dff
#from modules.ReadResults import read_neural_trials
from modules.ReadResults import read_move_offset
from modules.ReadResults import read_significance
from modules.ReadResults import read_bpod_mat_data
from modules.ReadResults import read_ROI_label
from plot import Basic_alignments
from plot import calcium_transient
import h5py
from modules.fig1_mask import plotter_all_masks
# %%
def read_neural_trials(ops):
    # read h5 file.
    f = h5py.File(
        os.path.join(ops['save_path0'], 'neural_trials.h5'),
        'r')
    neural_trials = dict()
    for trial in f['trial_id'].keys():
        neural_trials[trial] = dict()
        for data in f['trial_id'][trial].keys():
            neural_trials[trial][data] = np.array(f['trial_id'][trial][data])
    f.close()
    return neural_trials
# %%
def find_data(neural_trials):
    trial_type = []
    align_data = []
    min_len = 125+30
    
    for trial in neural_trials.keys():
        trial_type.append(neural_trials[trial]['trial_types'].item())
        min_len = min(len(neural_trials[trial]['time']),min_len)
        align_data.append(neural_trials[trial]['dff'][:,:min_len])
    align_time = neural_trials['10']['time']-neural_trials['10']['time_start']
    align_time = align_time[:min_len]
    return align_data, align_time, trial_type
def find_data_sig(neural_trials, labels):
    trial_type = []
    align_data = []
    min_len = 125+30
    
    for trial in neural_trials.keys():
        trial_type.append(neural_trials[trial]['trial_types'].item())
        min_len = min(len(neural_trials[trial]['time']),min_len)
        align_data.append(neural_trials[trial]['dff'][labels == 1,:min_len])
    align_time = neural_trials['10']['time']-neural_trials['10']['time_start']
    align_time = align_time[:min_len]
    return align_data, align_time, trial_type
def plot_2p(axs, align_data, align_time, trial_type, type_plot = np.nan):
    if np.isnan(type_plot):
        align_data_1 = np.concatenate(align_data , axis = 0)
    else:
        align_data_temp = [align_data[i] for i in range(len(trial_type)) if trial_type[i] == type_plot]
        num_trials = len(align_data_temp)
        num_neurons = len(align_data_temp[0])
        align_data_1 = np.concatenate(align_data_temp , axis = 0)
        align_data_1 = np.array(align_data_1)
    
    trajectory_mean = np.mean(align_data_1 , axis = 0)
    
    trajectory_sem = np.std(align_data_1 , axis = 0)/np.sqrt(len(align_data_1))
    #trajectory_sem = np.std(align_data_1 , axis = 0)
    color_tag = 'k'
    if type_plot ==0:
        color_tag = 'k'
    elif type_plot == 1:
        color_tag = 'red'
    axs.fill_between(align_time/1000 ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = color_tag , alpha = 0.3)
    #print(len(trajectory_mean))
    axs.plot(align_time/1000 , trajectory_mean , color = color_tag, linewidth = 1, label = str(num_trials) + ' trial, '
                     + str(num_neurons) +' neuron')
    #axs.axvline(x = 0, color = 'gray', linestyle='--')
    start_stim = 0
    end_stim = 5
    if type_plot == 1:
        print(np.nanmax(np.abs(np.diff(trajectory_mean))))
        start_stim = np.where(np.abs(np.diff(trajectory_mean))>0.1)[0][0]
        end_stim = np.where(np.abs(np.diff(trajectory_mean))>0.1)[0][-1]
        axs.axvspan(align_time[start_stim]/1000, align_time[end_stim]/1000, alpha=0.1, color='red')
    #axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Time  (s)')
    axs.set_ylabel('df/f (z-scored)')
    axs.set_xlim([-1,4])
    axs.set_ylim([-0.07,0.75])
    axs.legend()
    return start_stim, end_stim
    
def plot_heatmap_neuron(ax, end_stim, neu_seq_raw, neu_time, trial_type, type_plot = np.nan, sort = 1, sort_idx_neu = np.nan):
    if np.isnan(type_plot):
        neu_seq = neu_seq_raw
        ax.set_title('All trials Heatmap')
    else:
        neu_seq = [neu_seq_raw[i] for i in range(len(trial_type)) if trial_type[i] == type_plot]
        neu_seq = np.array(neu_seq)
        if type_plot == 1:
            ax.set_title('Opto trials Heatmap')
        else:
            ax.set_title('Control trials Heatmap')
        
        
    if not np.isnan(np.nansum(neu_seq)) and len(neu_seq)>0:
        mean = np.nanmean(neu_seq, axis=0)
        smoothed_mean = np.array([np.convolve(row, np.ones(5)/5, mode='same') for row in mean])
        
        if sort == 1:
            #sort_idx_neu = np.argsort(ratio)
            # print(smoothed_mean[:,end_stim:].shape)
            # print(smoothed_mean.shape)
            # print(end_stim)
            sort_idx_neu = np.argmax(smoothed_mean[:,end_stim:], axis=1).reshape(-1).argsort()
            
            mean = mean[sort_idx_neu,:].copy()
            ax.set_ylabel('neuron id (sorted)')
        elif sort == 2:
            mean = mean[sort_idx_neu,:].copy()
            ax.set_ylabel('neuron id (sorted by opto acitivity)')
        else:
            ax.set_ylabel('neuron id (not sorted)')
        
        cmap = plt.cm.inferno
        im = ax.imshow(mean, interpolation='nearest', aspect='auto', vmin = -0.15, vmax = 0.5, cmap=cmap, extent = [np.min(neu_time)/1000 , np.max(neu_time)/1000, 0 , int(neu_seq.shape[1])])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel('Time (s)')
        ax.axvline(0, color='black', lw=1, label='stim', linestyle='--')
        ax.set_yticks([0, int(neu_seq.shape[1]/3), int(neu_seq.shape[1]*2/3), int(neu_seq.shape[1])])
        ax.set_yticklabels([0, int(neu_seq.shape[1]/3), int(neu_seq.shape[1]*2/3), int(neu_seq.shape[1])])
        plt.colorbar(im , ax = ax)
        
        ax.set_xlim([-1,4])
    return sort_idx_neu

def find_effective(end_stim, neu_seq_raw, neu_time, trial_type):
    win = 10
    neu_seq_control = [neu_seq_raw[i] for i in range(len(trial_type)) if trial_type[i] == 0]
    neu_seq_opto = [neu_seq_raw[i] for i in range(len(trial_type)) if trial_type[i] == 0]
    neu_seq_control = np.array(neu_seq_control)
    neu_seq_opto = np.array(neu_seq_opto)
    mean_control = np.nanmean(neu_seq_control, axis=0)
    mean_opto = np.nanmean(neu_seq_opto, axis=0)
    labels = np.zeros(len(mean_opto))
    for i in range(len(mean_opto)):
        # if np.abs(np.mean(mean_control[i, end_stim+1:end_stim+1+win]) - np.mean(mean_opto[i, end_stim:end_stim+win])) > 0.01:
        #     labels[i] = -1
        #print(np.max(mean_opto[i, end_stim:end_stim+win]))
        if np.max(mean_opto[i, end_stim:end_stim+win]) > 0.1:
            labels[i] = 1
    return labels
def find_effective_basic(sort_idx_neu):
    labels = np.zeros(len(sort_idx_neu))
    for i in sort_idx_neu[:160]:
        labels[i] = 1
    return labels

# %% loading data
############## reading neural data
cate_delay = 25
data_date ='20250617'
session_data_path = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/2p imaging/Data/SA09_LG/SA09_'+data_date
ops = np.load(
    os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
    allow_pickle=True).item()

ops['save_path0'] = os.path.join(session_data_path)

# %% outpUt location
subject = 'SA09_LG'
output_dir_onedrive = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/2p imaging/Figures/' + subject + '/' + data_date + '/'
output_dir_local = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/2p imaging/Figures/' + subject + '/' + data_date + '/'
last_day = '20220617'
#Trialization_Opto.run(ops)
neural_trials = read_neural_trials(ops)
# %%

subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4 , 4, figure=fig)
[labels, masks, mean_func, max_func, mean_anat, masks_anat, masks_anat_corrected] = read_masks(ops)
plotter_masks = plotter_all_masks(labels, masks, mean_func, max_func, mean_anat, masks_anat)
mask_ax01 = plt.subplot(gs[0, 1])
plotter_masks.func(mask_ax01, 'mean')

align_data, align_time, trial_type = find_data(neural_trials)


plot_2p(plt.subplot(gs[0, 0]), align_data, align_time, trial_type, type_plot = 0)
start_stim, end_stim = plot_2p(plt.subplot(gs[0, 0]), align_data, align_time, trial_type, type_plot = 1)
plot_heatmap_neuron(plt.subplot(gs[1, 0]), end_stim, align_data, align_time, trial_type, type_plot = 0, sort = 0)
plot_heatmap_neuron(plt.subplot(gs[1, 1]), end_stim, align_data, align_time, trial_type, type_plot = 1, sort = 0)

plot_heatmap_neuron(plt.subplot(gs[2, 0]), end_stim, align_data, align_time, trial_type, type_plot = 0, sort = 1)
plot_heatmap_neuron(plt.subplot(gs[2, 1]), end_stim, align_data, align_time, trial_type, type_plot = 1, sort = 1)

sort_idx_neu = plot_heatmap_neuron(plt.subplot(gs[3, 1]), end_stim, align_data, align_time, trial_type, type_plot = 1, sort = 1)
plot_heatmap_neuron(plt.subplot(gs[3, 0]), end_stim, align_data, align_time, trial_type, type_plot = 0, sort = 2, sort_idx_neu = sort_idx_neu)

#labels = find_effective(end_stim, align_data, align_time, trial_type)
labels = find_effective_basic(sort_idx_neu)
align_data, align_time, trial_type =  find_data_sig(neural_trials, labels)

plotter_masks = plotter_all_masks(labels, masks, mean_func, max_func, mean_anat, masks_anat)
mask_ax01 = plt.subplot(gs[0, 3])
#plotter_masks.shared_masks(mask_ax01)
plotter_masks.func_effective(mask_ax01, 'mean')

plot_2p(plt.subplot(gs[0, 2]), align_data, align_time, trial_type, type_plot = 0)
start_stim, end_stim = plot_2p(plt.subplot(gs[0, 2]), align_data, align_time, trial_type, type_plot = 1)
plot_heatmap_neuron(plt.subplot(gs[1, 2]), end_stim, align_data, align_time, trial_type, type_plot = 0, sort = 0)
plot_heatmap_neuron(plt.subplot(gs[1, 3]), end_stim, align_data, align_time, trial_type, type_plot = 1, sort = 0)

plot_heatmap_neuron(plt.subplot(gs[2, 2]), end_stim, align_data, align_time, trial_type, type_plot = 0, sort = 1)
plot_heatmap_neuron(plt.subplot(gs[2, 3]), end_stim, align_data, align_time, trial_type, type_plot = 1, sort = 1)

sort_idx_neu_1 = plot_heatmap_neuron(plt.subplot(gs[3, 3]), end_stim, align_data, align_time, trial_type, type_plot = 1, sort = 1)
plot_heatmap_neuron(plt.subplot(gs[3, 2]), end_stim, align_data, align_time, trial_type, type_plot = 0, sort = 2, sort_idx_neu = sort_idx_neu_1)

plt.suptitle(subject + ' (' + data_date + ')')
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()

os.remove(fname)
subject_report.save(output_dir_onedrive + last_day  + '_alignment_10.pdf')
subject_report.close()
# %%

subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(3 , 3, figure=fig)
[labels, masks, mean_func, max_func, mean_anat, masks_anat, masks_anat_corrected] = read_masks(ops)
plotter_masks = plotter_all_masks(labels, masks, mean_func, max_func, mean_anat, masks_anat)
mask_ax01 = plt.subplot(gs[1, 1])
plotter_masks.func(mask_ax01, 'mean')

mask_ax01 = plt.subplot(gs[1, 0])
plotter_masks.func(mask_ax01, 'max', with_mask=False)

align_data, align_time, trial_type = find_data(neural_trials)

labels = find_effective_basic(sort_idx_neu)
align_data, align_time, trial_type =  find_data_sig(neural_trials, labels)

plotter_masks = plotter_all_masks(labels, masks, mean_func, max_func, mean_anat, masks_anat)
mask_ax01 = plt.subplot(gs[1, 2])
#plotter_masks.shared_masks(mask_ax01)
plotter_masks.func_effective(mask_ax01, 'mean')


plt.suptitle(subject + ' (' + data_date + ')')
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()

os.remove(fname)
subject_report.save(output_dir_onedrive + last_day  + '_mask_1.pdf')
subject_report.close()
# %%
fig = plt.figure(figsize=(105, 210))

gs = GridSpec(30, 15, figure=fig)
mask_ax01 = plt.subplot(gs[0:2, 0:2])
mask_ax02 = plt.subplot(gs[0:2, 2:4])
plotter_masks.func(mask_ax01, 'max')
plotter_masks.func_masks(mask_ax02)

# %%
labels_1 = find_effective(end_stim, align_data, align_time, trial_type)
print(np.sum(labels_1))