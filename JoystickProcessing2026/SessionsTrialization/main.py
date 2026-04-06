# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 19:46:42 2026

@author: saminnaji3
"""

import os
import numpy as np
from Modules import Trialization
from Modules import ReadResults

import warnings
warnings.filterwarnings("ignore")

# %%
subject = 'SA16_LG'
subject_id = 'SA16_'
date = '20260316/'
output_dir_onedrive = os.path.join('C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/2p imaging/Figures/', subject, 'all_figs', date)
output_dir_local = output_dir_onedrive
#C:\Users\Sana\OneDrive - Georgia Institute of Technology\Figure Joystick\Weekly_report_20260301\SA18_LG
initial_path = 'C:\\Users\\saminnaji3\\Downloads'

# # sa16
# data_dates = ['20251215', '20251226', '20251227', '20251228', '20251229', '20251230', '20260103', '20260105' , '20260106', '20260107']
# # sa18
# data_dates = ['20251226', '20251227', '20251228', '20251229', '20251230', '20260103', '20260104', '20260105' , '20260106', '20260107']
# # SA20
# data_dates = ['20260120', '20260121']
# # YH30
# data_dates = ['20260130']
# sa16
data_dates = ['20260105']
# %% reading data and trialization
list_session_data_path = []
for date in data_dates:
    list_session_data_path.append(os.path.join(initial_path, subject, subject_id + date))
  
for session in list_session_data_path:
    ops = np.load(
        os.path.join(session, 'suite2p', 'plane0','ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session)
    print('trialization of session: ' , session[-13:])
    Trialization.run(ops)
    
    
# %% try to read them
neural_trials = ReadResults.read_neural_trials(ops)
neural_data = ReadResults.read_neural_data(ops, 1)
