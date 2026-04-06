# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:06:53 2026

@author: saminnaji3
"""

import os
import fitz
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
import warnings

from Modules import ReadResults
from Modules import main_plots
from Modules import glm

warnings.filterwarnings("ignore")

# %%
subject = 'SA15_LG'
subject_id = 'SA15_'
date = '20260313/'
output_dir_onedrive = os.path.join('C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/2p imaging/Figures/', subject, 'all_figs', date)
output_dir_local = output_dir_onedrive
initial_path = 'C:\\Users\\saminnaji3\\Downloads\\passive'
# sa16
data_dates = ['20251215', '20251226', '20251227', '20251228', '20251229', '20251230', '20260103', '20260105' , '20260106', '20260107']
# sa18
data_dates = ['20251226', '20251227', '20251228', '20251229', '20251230', '20260103', '20260104', '20260105' , '20260106', '20260107']
# SA20
data_dates = ['20260120', '20260121']
# YH30
data_dates = ['20260130']

data_dates = ['20260217', '20260219', '20260220', '20260226', '20260227', '20260228', '20260301']
# %% reading data 
list_session_data_path = []
for date in data_dates:
    list_session_data_path.append(os.path.join(initial_path, subject, subject_id + date))
  
list_neural_data = []
labels_per_session = []
for session_data_path in list_session_data_path:
    ops = np.load(
        os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session_data_path)
    neural_data = ReadResults.read_neural_data(ops)
    list_neural_data.append(neural_data)
    labels = ReadResults.read_cluster_labels(ops)
    labels_per_session.append(labels)
    
# %% audio

list_session_data_path = []

subject = 'SA12_LG'
subject_id = 'SA12_'
data_dates = ['20251001', '20251002', '20251007', '20251008']
for date in data_dates:
    list_session_data_path.append(os.path.join(initial_path, subject, subject_id + date))
    
subject = 'SA13_LG'
subject_id = 'SA13_'
data_dates =['20250930', '20251001', '20251002', '20251003', '20251006', '20251007', '20251009']
for date in data_dates:
    list_session_data_path.append(os.path.join(initial_path, subject, subject_id + date))
  
list_neural_data = []
labels_per_session = []
for session_data_path in list_session_data_path:
    ops = np.load(
        os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session_data_path)
    neural_data = ReadResults.read_neural_data(ops)
    list_neural_data.append(neural_data)
    labels = ReadResults.read_cluster_labels(ops)
    labels_per_session.append(labels)
    
# %% video

list_session_data_path = []

subject = 'SA21_LG'
subject_id = 'SA21_'
data_dates = ['20260217', '20260219', '20260220', '20260226', '20260227', '20260228', '20260301']
for date in data_dates:
    list_session_data_path.append(os.path.join(initial_path, subject, subject_id + date))
    
subject = 'SA14_LG'
subject_id = 'SA14_'
data_dates = ['20251002', '20251003', '20251007', '20251008', '20251009']
for date in data_dates:
    list_session_data_path.append(os.path.join(initial_path, subject, subject_id + date))
    
subject = 'SA15_LG'
subject_id = 'SA15_'
data_dates = ['20251002', '20251003', '20251009']
for date in data_dates:
    list_session_data_path.append(os.path.join(initial_path, subject, subject_id + date))
  
list_neural_data = []
labels_per_session = []
for session_data_path in list_session_data_path:
    ops = np.load(
        os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session_data_path)
    neural_data = ReadResults.read_neural_data(ops)
    # --- ISI CALCULATION LOGIC ---
    stim_labels = neural_data['stim_labels']
    onsets = stim_labels[:, 0]  # Column 0 is trigger time (ms or s)
    
    # Calculate differences between consecutive onsets
    intervals = np.diff(onsets)
    
    # 1. Preceding ISI: [NaN, onset1-onset0, onset2-onset1, ...]
    pre_isi = np.concatenate([[np.nan], intervals])
    
    # 2. Next ISI: [onset1-onset0, onset2-onset1, ..., NaN]
    next_isi = np.concatenate([intervals, [np.nan]])
    
    # Append these as new columns (horizontally stack)
    # This adds them to the end of the existing columns
    updated_labels = np.column_stack((stim_labels, pre_isi, next_isi))
    
    # Update the dictionary
    neural_data['stim_labels'] = updated_labels
    # -----------------------------
    list_neural_data.append(neural_data)
    labels = ReadResults.read_cluster_labels(ops)
    labels_per_session.append(labels)

# %% plot the imaging field of view
main_plots.plot_fov_blank(list_session_data_path, output_dir_onedrive)
main_plots.plot_fov(list_session_data_path, output_dir_onedrive)

# %% this is to plot the aligned data
event_list = ['vis1', 'push1', 'retract1', 'vis2', 'push2', 'retract2', 'reward_delay', 'reward', 'trial_end']
pre_list = [10, 45, 45, 45, 45, 45, 45, 45, 60]
post_list = [45, 45, 45, 45, 45, 45, 45, 45, 1]
my_conditions = {'outcome': 1}
master_pdf_name = 'all_sessions_analysis.pdf'
existing_pdf_path = os.path.join(output_dir_onedrive, master_pdf_name)
main_plots.create_initial_pdf(output_dir_onedrive, filename = master_pdf_name)
main_plots.make_data_final(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, labels_per_session, grant = 0, 
                           conditions=my_conditions, plot_mode = 'pool', target_cluster = None,
                           existing_pdf_path = existing_pdf_path)
main_plots.make_data_final(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, labels_per_session, grant = 1, 
                           conditions=my_conditions, plot_mode = 'separate', target_cluster = None,
                           existing_pdf_path = existing_pdf_path)
for i in np.unique(np.concatenate(labels_per_session)):

    main_plots.make_data_final(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, labels_per_session, grant = 0, 
                               conditions=my_conditions, plot_mode = 'specific', target_cluster = i,
                               existing_pdf_path = existing_pdf_path)
    
# %%
# Example: Select only Oddball stims (column 5 == 1) that are NOT sequence -1
my_conditions = {
    
}
pre_list = [180, 45, 45, 45, 45, 45, 45, 45, 60]
post_list = [45, 45, 45, 45, 45, 45, 45, 45, 1]
main_plots.make_data_final(
    output_dir_onedrive=output_dir_onedrive,
    list_neural_data=list_neural_data,
    event_list=['stim'],  # Trigger the stim logic
    conditions=my_conditions,
    pre_list= pre_list, post_list= post_list, list_labels = labels_per_session, grant = 1)
    
# %%
# Define conditions for 3 columns
master_pdf_name = 'all_sessions_analysis_separate_fix_jitter.pdf'
existing_pdf_path = os.path.join(output_dir_onedrive, master_pdf_name)
main_plots.create_initial_pdf(output_dir_onedrive, filename = master_pdf_name)

column_conds = [
    {'seq_num': 0, 'jitter': 0}, # Column 1: Standard Image 1
    {'seq_num': 1, 'jitter': 0}, # Column 2: Oddball Image 1
    {'seq_num': 2, 'jitter': 0},                 # Column 3: Jittered stimuli
    {'seq_num': 3, 'jitter': 0}                 # Column 3: Jittered stimuli
]


pre_list = [120, 90, 90, 180]
post_list = [180, 90, 90, 120]

main_plots.make_data_final_new(
    output_dir_onedrive=output_dir_onedrive,
    list_neural_data=list_neural_data,
    list_labels=labels_per_session,
    pre_list=pre_list, 
    post_list=post_list,
    column_conditions=column_conds, # New Argument
    grant=1, 
    plot_mode='separate',
    existing_pdf_path=existing_pdf_path
)

column_conds = [
    {'seq_num': 0, 'jitter': 1}, # Column 1: Standard Image 1
    {'seq_num': 1, 'jitter': 1}, # Column 2: Oddball Image 1
    {'seq_num': 2, 'jitter': 1},                 # Column 3: Jittered stimuli
    {'seq_num': 3, 'jitter': 1}                 # Column 3: Jittered stimuli
]
pre_list = [120, 90, 90, 60]
post_list = [180, 90, 90, 120]
main_plots.make_data_final_new(
    output_dir_onedrive=output_dir_onedrive,
    list_neural_data=list_neural_data,
    list_labels=labels_per_session,
    pre_list=pre_list, 
    post_list=post_list,
    column_conditions=column_conds, # New Argument
    grant=1, 
    plot_mode='separate',
    existing_pdf_path=existing_pdf_path
)
# %%
# Solid lines conditions
# Define conditions for 3 columns
master_pdf_name = 'all_sessions_analysis_comparison_oddball_short.pdf'
existing_pdf_path = os.path.join(output_dir_onedrive, master_pdf_name)
main_plots.create_initial_pdf(output_dir_onedrive, filename = master_pdf_name)

conds_solid = [
    {'seq_num': 0, 'jitter': 0, 'img_seq': [1, 2, 3, 4, 5]}, # Column 1: Standard Image 1
    {'seq_num': 1, 'jitter': 0, 'img_seq': [1, 2, 3, 4, 5]}, # Column 2: Oddball Image 1
    {'seq_num': 2, 'jitter': 0, 'img_seq': [1, 2, 3, 4, 5]},                 # Column 3: Jittered stimuli
    {'seq_num': 3, 'jitter': 0, 'img_seq': [1, 2, 3, 4, 5]}                 # Column 3: Jittered stimuli
]

# Dashed lines conditions
conds_dashed = [
    {'seq_num': 0, 'jitter': 0, 'img_seq': [-1], 'oddball': 0}, # Column 1: Standard Image 1
    {'seq_num': 1, 'jitter': 0, 'img_seq': [-1], 'oddball': 0}, # Column 2: Oddball Image 1
    {'seq_num': 2, 'jitter': 0, 'img_seq': [-1], 'oddball': 0},                 # Column 3: Jittered stimuli
    {'seq_num': 3, 'jitter': 0, 'img_seq': [-1], 'oddball': 0}                 # Column 3: Jittered stimuli
]
pre_list = [120, 120, 90, 180]
post_list = [180, 180, 90, 120]

main_plots.make_data_comparison(
    
    output_dir_onedrive=output_dir_onedrive,
    list_neural_data=list_neural_data,
    pre_list=pre_list, 
    post_list=post_list,
    list_labels=labels_per_session,
    column_conditions1=conds_solid,
    column_conditions2=conds_dashed,
    grant=1,
    plot_mode='specific',
    target_cluster = -1,
    existing_pdf_path=existing_pdf_path
)

main_plots.make_data_comparison(
    
    output_dir_onedrive=output_dir_onedrive,
    list_neural_data=list_neural_data,
    pre_list=pre_list, 
    post_list=post_list,
    list_labels=labels_per_session,
    column_conditions1=conds_solid,
    column_conditions2=conds_dashed,
    grant=1,
    plot_mode='specific',
    target_cluster = 0,
    existing_pdf_path=existing_pdf_path
)

main_plots.make_data_comparison(
    
    output_dir_onedrive=output_dir_onedrive,
    list_neural_data=list_neural_data,
    pre_list=pre_list, 
    post_list=post_list,
    list_labels=labels_per_session,
    column_conditions1=conds_solid,
    column_conditions2=conds_dashed,
    grant=1,
    plot_mode='specific',
    target_cluster = 1,
    existing_pdf_path=existing_pdf_path
)
# %%
# Solid lines conditions
# Define conditions for 3 columns
master_pdf_name = 'all_sessions_analysis_comparison_fix_jiter_test.pdf'
existing_pdf_path = os.path.join(output_dir_onedrive, master_pdf_name)
main_plots.create_initial_pdf(output_dir_onedrive, filename = master_pdf_name)

conds_solid = [
    {'seq_num': 0, 'jitter': 0, 'img_seq': [1, 2, 3, 4, 5]}, # Column 1: Standard Image 1
    {'seq_num': 1, 'jitter': 0, 'img_seq': [1, 2, 3, 4, 5]}, # Column 2: Oddball Image 1
    {'seq_num': 2, 'jitter': 0, 'img_seq': [1, 2, 3, 4, 5]},                 # Column 3: Jittered stimuli
    {'seq_num': 3, 'jitter': 0, 'img_seq': [1, 2, 3, 4, 5]}                 # Column 3: Jittered stimuli
]

# Dashed lines conditions
conds_dashed = [
    {'seq_num': 0, 'jitter': 1, 'pre_isi': '>1500'}, # Column 1: Standard Image 1
    {'seq_num': 1, 'jitter': 1, 'pre_isi': '>1500'}, # Column 2: Oddball Image 1
    {'seq_num': 2, 'jitter': 1, 'pre_isi': '>1500'},                 # Column 3: Jittered stimuli
    {'seq_num': 3, 'jitter': 1, 'pre_isi': '>1500'}                 # Column 3: Jittered stimuli
]
pre_list = [120, 90, 90, 90]
post_list = [180, 90, 90, 120]

main_plots.make_data_comparison(
    
    output_dir_onedrive=output_dir_onedrive,
    list_neural_data=list_neural_data,
    pre_list=pre_list, 
    post_list=post_list,
    list_labels=labels_per_session,
    column_conditions1=conds_solid,
    column_conditions2=conds_dashed,
    grant=1,
    plot_mode='specific',
    target_cluster = -1,
    existing_pdf_path=existing_pdf_path
)

main_plots.make_data_comparison(
    
    output_dir_onedrive=output_dir_onedrive,
    list_neural_data=list_neural_data,
    pre_list=pre_list, 
    post_list=post_list,
    list_labels=labels_per_session,
    column_conditions1=conds_solid,
    column_conditions2=conds_dashed,
    grant=1,
    plot_mode='specific',
    target_cluster = 0,
    existing_pdf_path=existing_pdf_path
)

main_plots.make_data_comparison(
    
    output_dir_onedrive=output_dir_onedrive,
    list_neural_data=list_neural_data,
    pre_list=pre_list, 
    post_list=post_list,
    list_labels=labels_per_session,
    column_conditions1=conds_solid,
    column_conditions2=conds_dashed,
    grant=1,
    plot_mode='specific',
    target_cluster = 1,
    existing_pdf_path=existing_pdf_path
)
