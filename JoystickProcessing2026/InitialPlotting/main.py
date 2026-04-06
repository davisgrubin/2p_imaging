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
subject = 'SA16_LG'
subject_id = 'SA16_'
date = '20260330/'
output_dir_onedrive = os.path.join('C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/2p imaging/Figures/', subject, 'all_figs', date)
output_dir_local = output_dir_onedrive
initial_path = 'C:\\Users\\saminnaji3\\Downloads'
# sa16
data_dates = ['20251215', '20251226', '20251227', '20251228', '20251229', '20251230', '20260103', '20260105' , '20260106', '20260107']
# sa18
data_dates = ['20251226', '20251227', '20251228', '20251229', '20251230', '20260103', '20260104', '20260105' , '20260106', '20260107']
# SA20
data_dates = ['20260120', '20260121']
# YH30
data_dates = ['20260130']
# sa18
data_dates = ['20260121', '20260122']
data_dates = ['20260105']
# %% reading data 
list_session_data_path = []
for date in data_dates:
    list_session_data_path.append(os.path.join(initial_path, subject, subject_id + date))
  
list_neural_data = []
all_weights = []
for session_data_path in list_session_data_path:
    ops = np.load(
        os.path.join(session_data_path,'ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session_data_path)
    neural_data = ReadResults.read_neural_data(ops)
    list_neural_data.append(neural_data)
    glm_input = glm.create_glm_input(neural_data)
    weights = glm.compute_glm_weights(glm_input, neural_data['dff'], ops, alpha=1)
    all_weights.append(weights)
    glm.save_weights(ops, weights)
    
labels_per_session = glm.clustering(all_weights, n_clusters = 7)
for num_sess, session_data_path in enumerate(list_session_data_path):
    ops = np.load(
        os.path.join(session_data_path,'ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session_data_path)
    glm.save_cluster_labels(ops, labels_per_session[num_sess])
# %% plot the imaging field of view
main_plots.plot_fov_blank(list_session_data_path, output_dir_onedrive)
main_plots.plot_fov(list_session_data_path, output_dir_onedrive)

# %% this is to plot the aligned data
event_list = ['vis1', 'push1', 'retract1', 'vis2', 'push2', 'retract2', 'reward_delay', 'reward', 'trial_end']
pre_list = [10, 45, 45, 45, 45, 45, 45, 45, 60]
post_list = [45, 45, 45, 45, 45, 45, 45, 45, 1]
my_conditions = {'outcome': 1}
master_pdf_name = 'all_sessions_analysis_new2.pdf'
existing_pdf_path = os.path.join(output_dir_onedrive, master_pdf_name)
main_plots.create_initial_pdf(output_dir_onedrive, filename = master_pdf_name)
# main_plots.make_data_final(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, labels_per_session, grant = 0, 
#                            conditions=my_conditions, plot_mode = 'pool', target_cluster = None,
#                            existing_pdf_path = existing_pdf_path)
main_plots.make_data_final(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, labels_per_session, grant = 0, 
                           conditions=my_conditions, plot_mode = 'separate', target_cluster = None,
                           existing_pdf_path = existing_pdf_path)

my_conditions = {'outcome': 2}

main_plots.make_data_final(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, labels_per_session, grant = 0, 
                           conditions=my_conditions, plot_mode = 'separate', target_cluster = None,
                           existing_pdf_path = existing_pdf_path)

my_conditions = {'outcome': 3}

main_plots.make_data_final(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, labels_per_session, grant = 0, 
                           conditions=my_conditions, plot_mode = 'separate', target_cluster = None,
                           existing_pdf_path = existing_pdf_path)

my_conditions = {'outcome': 4}

main_plots.make_data_final(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, labels_per_session, grant = 0, 
                           conditions=my_conditions, plot_mode = 'separate', target_cluster = None,
                           existing_pdf_path = existing_pdf_path)

my_conditions = {'outcome': 5}
master_pdf_name = 'all_sessions_analysis_long.pdf'

main_plots.make_data_final(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, labels_per_session, grant = 0, 
                           conditions=my_conditions, plot_mode = 'separate', target_cluster = None,
                           existing_pdf_path = existing_pdf_path)


my_conditions = {'outcome': 6}

main_plots.make_data_final(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, labels_per_session, grant = 0, 
                           conditions=my_conditions, plot_mode = 'separate', target_cluster = None,
                           existing_pdf_path = existing_pdf_path)

# %%
plt.plot(np.diff(list_neural_data[0]['time']))
plt.show()
# %%
event_list = ['vis1', 'push1', 'retract1', 'vis2', 'push2', 'retract2', 'reward_delay', 'reward', 'trial_end', 'vis_stim']
pre_list = [10, 45, 45, 45, 45, 45, 45, 45, 60, 45]
post_list = [45, 45, 45, 45, 45, 45, 45, 45, 1, 45]
my_conditions = {'outcome': 1}
master_pdf_name = 'all_sessions_analysis_with_pupil.pdf'
existing_pdf_path = os.path.join(output_dir_onedrive, master_pdf_name)
main_plots.create_initial_pdf(output_dir_onedrive, filename = master_pdf_name)

main_plots.make_data_final_with_pupil(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, labels_per_session, grant = 1, 
                           conditions=None, plot_mode = 'separate', target_cluster = None,
                           existing_pdf_path = existing_pdf_path)
