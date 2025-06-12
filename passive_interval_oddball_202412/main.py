#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import gc
import os
import argparse
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
import visualization2_3331Random
import visualization3_1451ShortLong
import visualization4_4131FixJitterOdd
from webpage import pack_webpage_main

def run(session_config_list):
    smooth = True

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
    for i in range(len(list_ops)):
        clean_memap_path(list_ops[i])

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
    #fn1 = visualization1_FieldOfView.run(session_config_list, smooth)
    fn1 = []

    print('===============================================')
    print('========= plotting 3331Random results =========')
    print('===============================================')
    #fn2 = visualization2_3331Random.run(session_config_list, smooth)
    fn2 = []

    print('===============================================')
    print('======= plotting 1451ShortLong results ========')
    print('===============================================')
    #fn3 = visualization3_1451ShortLong.run(session_config_list, smooth)
    fn3 = []

    print('===============================================')
    print('====== plotting 4131FixJitterOdd results ======')
    print('===============================================')
    fn4 = visualization4_4131FixJitterOdd.run(session_config_list, smooth)
    #fn4 = []

    print('===============================================')
    print('============ saving session report ============')
    print('===============================================')
    print('Saving results')
    pack_webpage_main.run(
        session_config_list,
        [fn1, fn2, fn3, fn4],
        ['Field of View', 'The Random Session', 'The Short-Long Session', 'The Fix-Jitter-Oddball Session'],
        ['random', 'fix_jitter_odd', 'short_long'])
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
    COMMANDLINE_MODE = False
    from session_configs import all_config_list
    
    if COMMANDLINE_MODE:
        parser = argparse.ArgumentParser(description='Experiments can go shit but Yicong will love you forever!')
        parser.add_argument('--config_list', required=True, type=str, help='Whether run denoising.')
        args = parser.parse_args()
        for subject, session_config_list in zip(
                ['YH01VT', 'YH02VT', 'YH03VT', 'YH14SC', 'YH16SC',
                 'YH17VT', 'YH18VT', 'YH19VT', 'YH20SC', 'YH21SC',
                 'PPC', 'V1'],
                all_config_list
            ):
            if subject in args.config_list:
                run(session_config_list)
                
    else:
        session_config_test = {
            'list_session_name' : {
                #'VTYH01_PPC_20250106_3331Random' : 'random',
                #'VTYH01_PPC_20250107_3331Random' : 'random',
                #'VTYH01_PPC_20250108_3331Random' : 'random',
                #'VTYH02_PPC_20250108_3331Random' : 'random',
                #'VTYH02_PPC_20250109_3331Random' : 'random',
                #'VTYH02_PPC_20250111_3331Random' : 'random',
                #'VTYH03_PPC_20250106_3331Random' : 'random',
                #'VTYH03_PPC_20250107_3331Random' : 'random',
                #'VTYH03_PPC_20250108_3331Random' : 'random',
                'VTYH01_PPC_20250201_4131FixJitterOdd' : 'fix_jitter_odd',
                'VTYH01_PPC_20250203_4131FixJitterOdd' : 'fix_jitter_odd',
                'VTYH01_PPC_20250204_4131FixJitterOdd' : 'fix_jitter_odd',
                #'VTYH02_PPC_20250121_4131FixJitterOdd' : 'fix_jitter_odd',
                #'VTYH02_PPC_20250202_4131FixJitterOdd' : 'fix_jitter_odd',
                #'VTYH02_PPC_20250203_4131FixJitterOdd' : 'fix_jitter_odd',
                #'VTYH03_PPC_20250131_4131FixJitterOdd' : 'fix_jitter_odd',
                #'VTYH03_PPC_20250201_4131FixJitterOdd' : 'fix_jitter_odd',
                #'VTYH03_PPC_20250203_4131FixJitterOdd' : 'fix_jitter_odd',
                #'VTYH01_PPC_20250225_1451ShortLong' : 'short_long',
                #'VTYH01_PPC_20250226_1451ShortLong' : 'short_long',
                #'VTYH01_PPC_20250228_1451ShortLong' : 'short_long',
                #'VTYH02_PPC_20250225_1451ShortLong' : 'short_long',
                #'VTYH02_PPC_20250226_1415ShortLong' : 'short_long',
                #'VTYH02_PPC_20250228_1451ShortLong' : 'short_long',
                #'VTYH03_PPC_20250218_1451ShortLong' : 'short_long',
                #'VTYH03_PPC_20250219_1451ShortLong' : 'short_long',
                #'VTYH03_PPC_20250221_1451ShortLong' : 'short_long',
                },
            'session_folder' : 'test',
            'sig_tag' : 'all',
            'force_label' : None,
            }
        session_config_list_test = {
            'list_config': [
                session_config_test,
                ],
            'label_names' : {
                '-1':'Exc',
                '1':'Inh_VIP',
                '2':'Inh_SST',
                },
            'subject_name' : 'YH19VT',
            'output_filename' : 'test_passive.html'
            }
    
        '''
        
        # run(session_config_list_test)
        
        import matplotlib.pyplot as plt
        session_config_list = combine_session_config_list(session_config_list_test)
        list_ops = read_ops(session_config_list['list_session_data_path'])
        ops = list_ops[0]
        
        from modules.ReadResults import read_all
        [list_labels, list_masks,
         list_neural_trials, list_move_offset, list_significance
         ] = read_all(session_config_list, smooth=False)
        neural_trials = list_neural_trials[0]
        dff = neural_trials['dff']
        label_names = {'-1':'Exc', '1':'Inh_VIP', '2':'Inh_SST'}
        cate = [-1,1,2]
        roi_id = None
        norm_mode='none'
       
        cluster_cmap = plt.cm.hsv
        standard = 1
        oddball = 1
        block = 0
        mode = 'common'
        temp_folder = 'temp_'+session_config_list['subject_name']
        if not os.path.exists(os.path.join('results', 'temp_'+session_config_list['subject_name'])):
            os.makedirs(os.path.join('results', 'temp_'+session_config_list['subject_name']))
        if not os.path.exists(os.path.join('results', 'temp_'+session_config_list['subject_name'], 'alignment_memmap')):
            os.makedirs(os.path.join('results', 'temp_'+session_config_list['subject_name'], 'alignment_memmap'))
        
        '''
