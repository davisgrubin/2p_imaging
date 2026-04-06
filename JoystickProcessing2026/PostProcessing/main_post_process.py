# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 16:41:30 2026

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

manual = True

data_date ='20260111'
subject_name = 'SA18'
subject = subject_name + '_LG'

session_data_path = 'C:\\Users\\saminnaji3\\Downloads\\' + subject + '\\' + subject_name + '_' + data_date
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
    run_manual_postprocess.run(session_data_path)

else:
    print('================First Round of Quality Control===================')
    run_postprocess.run(session_data_path)
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