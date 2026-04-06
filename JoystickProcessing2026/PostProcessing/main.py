# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:44:23 2025

@author: saminnaji3
"""

import os
import run_postprocess
import run_manual_postprocess
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules.ReadResults import read_dff
from modules.ReadResults import save_lables

need_label = 0 
manual = True
data_date ='20260216'
subject_num = 'SA20'
subject = 'SA20_WT'
#session_data_path = 'C:\\Users\\saminnaji3\\OneDrive - Georgia Institute of Technology\\2p imaging\\Data\\' + subject+ '\\' + subject_num + '_' + data_date
session_data_path = 'C:\\Users\\saminnaji3\\Downloads\\' + subject+ '\\' + subject_num + '_' + data_date
session_data_path = 'C:\\Users\\saminnaji3\\Downloads\\SA20_20260216'
output_dir_onedrive = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/2p imaging/Figures/' + subject + '/RawTraces/All_ROIs/'

ops = np.load(
    os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
    allow_pickle=True).item()
ops['save_path0'] = os.path.join(session_data_path)


if manual == False:
    run_postprocess.run(session_data_path)
    dff = read_dff(ops, 'qc_results')
    good_roi = np.arange(dff.shape[0])
    bad_roi = []
    save_lables(ops, good_roi, bad_roi)

if manual == True:
    if  os.path.exists(session_data_path + '\\ROI_label.h5') and os.path.isfile(session_data_path + '\\ROI_label.h5'):
        run_manual_postprocess.run(session_data_path)
    else:
        print('================First Round of Quality Control=====================')
        run_postprocess.run(session_data_path)
        need_label = 1
        dff = read_dff(ops, 'qc_results')
        fs = 30
        time_min = list(np.arange(0 , dff.shape[1])/(fs*60))
        time_s = np.arange(0 , dff.shape[1])/(fs)
        total_neurons = dff.shape[0]
        
        start = 0
        for itr in range(1, total_neurons//100+2):
            end = min(100*itr, total_neurons)
            curr_max = 0 
            shift = np.nanmin(dff[0, :])
        
            for i in range(start, end):
                curr_dff = dff[i , :]
                median = np.nanmedian(curr_dff)
                curr_dff = curr_dff - median
                curr_min = np.nanmin(curr_dff)
                shift = shift + np.abs(curr_min) + curr_max
                curr_max = np.nanmax(curr_dff)
                if i == start:
                    fig = px.line(x = time_min[len(curr_dff)//2:len(curr_dff)//2+ 60*fs*3], y = list(curr_dff+shift)[len(curr_dff)//2:len(curr_dff)//2+ 60*fs*3] , labels = 'Neuron id= ' + str(i))
                else:
                    fig.add_trace(go.Scatter(x = time_min[len(curr_dff)//2:len(curr_dff)//2+ 60*fs*3], y = list(curr_dff+shift)[len(curr_dff)//2:len(curr_dff)//2+ 60*fs*3], mode='lines', name= 'Neuron id= ' + str(i)))
                    
            fig.update_layout(title='Raw dff traces', xaxis_title='Time (min)', yaxis_title='dff traces')
                  
            fig.write_html(output_dir_onedrive  + data_date + '_raw_traces_part' + str(itr) + '.html')
            start = end
    
        
    
# %%
# data_date ='20250821'
# subject_num = 'SA11'
# subject = 'SA11_LG'
# #session_data_path = 'C:\\Users\\saminnaji3\\OneDrive - Georgia Institute of Technology\\2p imaging\\Data\\' + subject+ '\\' + subject_num + '_' + data_date
# session_data_path = 'C:\\Users\\saminnaji3\\Downloads\\' + subject+ '\\' + subject_num + '_' + data_date
# output_dir_onedrive = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/2p imaging/Figures/' + subject + '/RawTraces/All_ROIs/'
need_plot = 0
need_num_ROI = 1
ops = np.load(
    os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
    allow_pickle=True).item()
ops['save_path0'] = os.path.join(session_data_path)
if need_label == 1:
    
    dff = read_dff(ops, 'qc_results')
    output_dir_onedrive = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/2p imaging/Figures/' + subject + '/RawTraces/Good_ROIs/'
    # if you want to mark good ROI write ids in good_roi else mark bad rois in bad_roi
    good_roi = []
    bad_roi = [53, 215]
    
    
    if len(bad_roi) == 0:
        for i in range(dff.shape[0]): 
            if not i in good_roi:
                bad_roi.append(i)
    elif len(good_roi) == 0:
        for i in range(dff.shape[0]):
            if not i in bad_roi:
                good_roi.append(i)
    # else:
    #     for i in range(300):
    #         if not i in bad_roi:
    #             good_roi.append(i)
    #     for i in range(dff.shape[0]): 
    #         if not i in good_roi and i not in bad_roi:
    #             bad_roi.append(i)
    #     print('error both good and bad have values')
    # for i in range(dff.shape[0]):
    #     if not i in bad_roi:
    #         good_roi.append(i)
    save_lables(ops, good_roi, bad_roi)
    
    
    run_manual_postprocess.run(session_data_path)
    
    if need_plot:
        dff = read_dff(ops, 'manual_qc_results')
        fs = 30
        time_min = np.arange(0 , dff.shape[1])/(fs*60)
        time_s = np.arange(0 , dff.shape[1])/(fs)
        
        total_neurons = dff.shape[0]
        start = 0
        for itr in range(1, total_neurons//150+2):
            end = min(150*itr, total_neurons)
            curr_max = 0 
            shift = np.nanmin(dff[0, :])
        
            for i in range(start, end):
                curr_dff = dff[i , :]
                median = np.nanmedian(curr_dff)
                curr_dff = curr_dff - median
                curr_min = np.nanmin(curr_dff)
                shift = shift + np.abs(curr_min) + curr_max
                curr_max = np.nanmax(curr_dff)
                if i == start:
                    fig = px.line(x = list(time_min), y = list(curr_dff+shift) , labels = 'Neuron id= ' + str(i))
                else:
                    fig.add_trace(go.Scatter(x = list(time_min), y = list(curr_dff+shift), mode='lines', name= 'Neuron id= ' + str(i)))
                    
            fig.update_layout(title='Raw dff traces', xaxis_title='Time (min)', yaxis_title='dff traces')
                  
            fig.write_html(output_dir_onedrive  + data_date + '_raw_traces_good_rois_part' + str(itr) + '.html')
            start = end
            
if need_num_ROI:
    dff = read_dff(ops, 'manual_qc_results')
    print('Number of good ROIs: ', dff.shape[0])

