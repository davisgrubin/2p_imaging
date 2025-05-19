#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import gc
import os
from datetime import datetime

from modules import Trialization
from modules import StatTest
from modules.ReadResults import read_ops
from modules.ReadResults import clean_memap_path

def combine_session_config_list(session_config_list):
    list_session_data_path = []
    for sc in session_config_list['list_config']:
        list_session_data_path += [
            os.path.join('results', sc['session_folder'], n)
            for n in sc['list_session_name'].keys()]
    list_session_name = [sc['list_session_name'] for sc in session_config_list['list_config']]
    list_session_name = {k: v for d in list_session_name for k, v in d.items()}
    session_config_list['list_session_name'] = list_session_name
    session_config_list['list_session_data_path'] = list_session_data_path
    return session_config_list

def get_roi_sign(significance, roi_id):
    r = significance['r_standard'][roi_id] +\
        significance['r_change'][roi_id] +\
        significance['r_oddball'][roi_id]
    return r

import visualization1_FieldOfView
import visualization2_Behavior
import visualization3_Perception
import visualization4_Decision
from webpage import pack_webpage_main

def run(session_config_list):
    smooth = False

    print('===============================================')
    print('============ Start Processing Data ============')
    print('===============================================')
    if not os.path.exists(os.path.join('results', 'temp_'+session_config_list['subject_name'])):
        os.makedirs(os.path.join('results', 'temp_'+session_config_list['subject_name']))
    if not os.path.exists(os.path.join('results', 'temp_'+session_config_list['subject_name'], 'alignment_memmap')):
        os.makedirs(os.path.join('results', 'temp_'+session_config_list['subject_name'], 'alignment_memmap'))
    print('Created canvas')
    session_config_list = combine_session_config_list(session_config_list)
    print('Processing {} sessions'.format(
        len(session_config_list['list_session_data_path'])))
    for n in session_config_list['list_session_name']:
        print(n)
    print('Reading ops.npy')
    list_ops = read_ops(session_config_list['list_session_data_path'])

    print('===============================================')
    print('============= trials segmentation =============')
    print('===============================================')
    for i in range(len(list_ops)):
        print('Trializing {}'.format(
            list(session_config_list['list_session_name'].keys())[i]))
        #Trialization.run(list_ops[i])

    print('===============================================')
    print('============== significance test ==============')
    print('===============================================')
    for i in range(len(list_ops)):
        print('Running significance test for {}'.format(
            list(session_config_list['list_session_name'].keys())[i]))
        #StatTest.run(list_ops[i])

    print('===============================================')
    print('======== plotting representative masks ========')
    print('===============================================')
    fn1 = visualization1_FieldOfView.run(session_config_list, smooth)
    #fn1 = []

    print('===============================================')
    print('========== plotting behavior results ==========')
    print('===============================================')
    fn2 = visualization2_Behavior.run(session_config_list, smooth)
    #fn2 = []
    
    print('===============================================')
    print('======== plotting perceptions results =========')
    print('===============================================')
    fn3 = visualization3_Perception.run(session_config_list, smooth)
    #fn3 = []
    
    print('===============================================')
    print('========== plotting decision results ==========')
    print('===============================================')
    fn4 = visualization4_Decision.run(session_config_list, smooth)
    #fn4 = []
    
    print('===============================================')
    print('============ saving session report ============')
    print('===============================================')
    print('Saving results')
    pack_webpage_main.run(
        session_config_list,
        [fn1, fn2, fn3, fn4],
        ['field of view', 'behavior', 'perception', 'decision'])
    for i in range(len(list_ops)):
        print('Cleaning memory mapping files for {}'.format(
            list(session_config_list['list_session_name'].keys())[i]))
        clean_memap_path(list_ops[i])
    print('Processing completed for all sessions')
    for n in session_config_list['list_session_name']:
        print(n)
    print('File saved as '+os.path.join('results', session_config_list['output_filename']))
    print('Finished at '+datetime.now().strftime("%Y.%m.%d %H:%M:%S"))
    del list_ops
    gc.collect()


if __name__ == "__main__":

    from session_configs import session_config_list_YH24LG

    for session_config_list in [
        session_config_list_YH24LG,
            ]:
        run(session_config_list)

    '''
    # run(session_config_list_YH24LG)

    session_config_list = combine_session_config_list(session_config_list_YH24LG)
    list_ops = read_ops(session_config_list['list_session_data_path'])
    ops = list_ops[0]
    from modules.ReadResults import read_all
    [list_labels, list_masks, 
     list_neural_trials, list_move_offset, list_significance
     ] = read_all(session_config_list, smooth=False)

    neural_trials = list_neural_trials[0]
    trial_labels = neural_trials['trial_labels']
    l_frames = 200
    r_frames = 200
    target_state = 'stim_seq'
    label_names = {'-1':'crux1'}
    cate = [-1]
    roi_id = None
    norm_mode='none'
    import matplotlib.pyplot as plt
    cluster_cmap = plt.cm.hsv
    temp_folder = 'temp_'+session_config_list['subject_name']
    if not os.path.exists(os.path.join('results', 'temp_'+session_config_list['subject_name'])):
        os.makedirs(os.path.join('results', 'temp_'+session_config_list['subject_name']))
    if not os.path.exists(os.path.join('results', 'temp_'+session_config_list['subject_name'], 'alignment_memmap')):
        os.makedirs(os.path.join('results', 'temp_'+session_config_list['subject_name'], 'alignment_memmap'))

    '''
