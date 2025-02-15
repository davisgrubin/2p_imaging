#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import os

from modules import Trialization
from modules import StatTest
from modules.ReadResults import read_ops
from modules.ReadResults import read_all

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

#%% main

def run(session_config_list):

    print('===============================================')
    print('============ Start Processing Data ============')
    print('===============================================')
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
        Trialization.run(list_ops[i])

    print('===============================================')
    print('============== significance test ==============')
    print('===============================================')
    for i in range(len(list_ops)):
        print('Running significance test for {}'.format(
            list(session_config_list['list_session_name'].keys())[i]))
        StatTest.run(list_ops[i])
    
    print('===============================================')
    print('============ reading saved results ============')
    print('===============================================')
    [list_labels, list_masks, list_vol, list_dff,
     list_neural_trials, list_move_offset, list_significance
     ] = read_all(session_config_list)
    print('Read {} session results'.format(len(list_ops)))

    print('===============================================')
    print('============= Start visualization =============')
    print('===============================================')
    if not os.path.exists(os.path.join('results', session_config_list['subject_name']+'_temp')):
        os.makedirs(os.path.join('results', session_config_list['subject_name']+'_temp'))
    print('Created canvas')

    print('===============================================')
    print('======== plotting representative masks ========')
    print('===============================================')
    fn1 = visualization1_FieldOfView.run(
        session_config_list,
        list_labels, list_masks, list_vol, list_dff, list_neural_trials, list_move_offset)

    print('===============================================')
    print('========= plotting 3331Random results =========')
    print('===============================================')
    fn2 = visualization2_3331Random.run(
        session_config_list, list_labels, list_vol, list_dff, list_neural_trials, list_significance)
    
    print('===============================================')
    print('======= plotting 1451ShortLong results ========')
    print('===============================================')
    fn3 = visualization3_1451ShortLong.run(
        session_config_list, list_labels, list_vol, list_dff, list_neural_trials, list_significance)

    print('===============================================')
    print('====== plotting 4131FixJitterOdd results ======')
    print('===============================================')
    fn4 = visualization4_4131FixJitterOdd.run(
        session_config_list, list_labels, list_vol, list_dff, list_neural_trials, list_significance)

    print('===============================================')
    print('============ saving session report ============')
    print('===============================================')
    print('Saving results')
    pack_webpage_main.run(session_config_list, fn1, fn2, fn3, fn4)
    for n in session_config_list['list_session_name']:
        print(n)
    print('Processing completed')
    print('File saved as '+os.path.join('results', session_config_list['output_filename']))


if __name__ == "__main__":

#%% subject configs

    # YH01VT.
    session_config_YH01VT = {
        'list_session_name' : {
            'VTYH01_PPC_20250106_3331Random' : 'random',
            'VTYH01_PPC_20250107_3331Random' : 'random',
            'VTYH01_PPC_20250108_3331Random' : 'random',
            'VTYH01_PPC_20250109_3331Random' : 'random',
            'VTYH01_PPC_20250111_3331Random' : 'random',
            'VTYH01_PPC_20250113_1451ShortLong' : 'short_long',
            'VTYH01_PPC_20250114_1451ShortLong' : 'short_long',
            'VTYH01_PPC_20250115_1451ShortLong' : 'short_long',
            'VTYH01_PPC_20250116_1451ShortLong' : 'short_long',
            'VTYH01_PPC_20250117_1451ShortLong' : 'short_long',
            'VTYH01_PPC_20250118_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH01_PPC_20250120_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH01_PPC_20250121_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH01_PPC_20250122_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH01_PPC_20250123_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH01_PPC_20250127_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH01_PPC_20250128_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH01_PPC_20250129_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH01_PPC_20250130_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH01_PPC_20250131_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH01_PPC_20250201_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH01_PPC_20250203_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH01_PPC_20250204_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH01_PPC_20250205_1451ShortLong' : 'short_long',
            'VTYH01_PPC_20250206_1451ShortLong' : 'short_long',
            'VTYH01_PPC_20250207_1451ShortLong' : 'short_long',
            'VTYH01_PPC_20250208_1451ShortLong' : 'short_long',
            'VTYH01_PPC_20250210_1451ShortLong' : 'short_long',
            'VTYH01_PPC_20250211_1451ShortLong' : 'short_long',
            'VTYH01_PPC_20250212_1451ShortLong' : 'short_long',
            },
        'session_folder' : 'YH01VT',
        'sig_tag' : 'all',
        'force_label' : None,
        }
    # YH02VT.
    session_config_YH02VT = {
        'list_session_name' : {
            'VTYH02_PPC_20250106_3331Random' : 'random',
            'VTYH02_PPC_20250107_3331Random' : 'random',
            'VTYH02_PPC_20250108_3331Random' : 'random',
            'VTYH02_PPC_20250109_3331Random' : 'random',
            'VTYH02_PPC_20250111_3331Random' : 'random',
            'VTYH02_PPC_20250113_1451ShortLong' : 'short_long',
            'VTYH02_PPC_20250114_1451ShortLong' : 'short_long',
            'VTYH02_PPC_20250115_1451ShortLong' : 'short_long',
            'VTYH02_PPC_20250116_1451ShortLong' : 'short_long',
            'VTYH02_PPC_20250117_1451ShortLong' : 'short_long',
            'VTYH02_PPC_20250118_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH02_PPC_20250120_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH02_PPC_20250121_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH02_PPC_20250122_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH02_PPC_20250123_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH02_PPC_20250127_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH02_PPC_20250129_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH02_PPC_20250130_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH02_PPC_20250131_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH02_PPC_20250202_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH02_PPC_20250203_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH02_PPC_20250205_1451ShortLong' : 'short_long',
            'VTYH02_PPC_20250206_1451ShortLong' : 'short_long',
            'VTYH02_PPC_20250207_1451ShortLong' : 'short_long',
            'VTYH02_PPC_20250208_1451ShortLong' : 'short_long',
            'VTYH02_PPC_20250210_1451ShortLong' : 'short_long',
            'VTYH02_PPC_20250211_1451ShortLong' : 'short_long',
            'VTYH02_PPC_20250212_1451ShortLong' : 'short_long',
            'VTYH02_PPC_20250213_1451ShortLong' : 'short_long',
            },
        'session_folder' : 'YH02VT',
        'sig_tag' : 'all',
        'force_label' : None,
        }
    # YH03VT.
    session_config_YH03VT = {
        'list_session_name' : {
            'VTYH03_PPC_20250106_3331Random' : 'random',
            'VTYH03_PPC_20250107_3331Random' : 'random',
            'VTYH03_PPC_20250108_3331Random' : 'random',
            'VTYH03_PPC_20250109_3331Random' : 'random',
            'VTYH03_PPC_20250111_3331Random' : 'random',
            'VTYH03_PPC_20250113_1451ShortLong' : 'short_long',
            'VTYH03_PPC_20250114_1451ShortLong' : 'short_long',
            'VTYH03_PPC_20250116_1451ShortLong' : 'short_long',
            'VTYH03_PPC_20250117_1451ShortLong' : 'short_long',
            'VTYH03_PPC_20250118_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH03_PPC_20250120_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH03_PPC_20250121_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH03_PPC_20250122_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH03_PPC_20250123_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH03_PPC_20250129_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH03_PPC_20250130_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH03_PPC_20250131_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH03_PPC_20250201_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH03_PPC_20250203_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH03_PPC_20250205_1451ShortLong' : 'short_long',
            'VTYH03_PPC_20250206_1451ShortLong' : 'short_long',
            'VTYH03_PPC_20250207_1451ShortLong' : 'short_long',
            'VTYH03_PPC_20250208_1451ShortLong' : 'short_long',
            'VTYH03_PPC_20250210_1451ShortLong' : 'short_long',
            'VTYH03_PPC_20250212_1451ShortLong' : 'short_long',
            },
        'session_folder' : 'YH03VT',
        'sig_tag' : 'all',
        'force_label' : None,
        }
    # YH07VT.
    session_config_YH07VT = {
        'list_session_name' : {
            'VTYH07_PPC_20250131_3331Random' : 'random',
            },
        'session_folder' : 'YH07VT',
        'sig_tag' : 'all',
        'force_label' : 1,
        }

#%% list configs

    session_config_list_YH01VT = {
        'list_config': [
            session_config_YH01VT,
            ],
        'label_names' : {
            '-1':'Exc',
            '1':'Inh_VIP',
            '2':'Inh_SST',
            },
        'subject_name' : 'YH01VT',
        'output_filename' : 'YH01VT_PPC_passive.html'
        }
    session_config_list_YH02VT = {
        'list_config': [
            session_config_YH02VT,
            ],
        'label_names' : {
            '-1':'Exc',
            '1':'Inh_VIP',
            '2':'Inh_SST',
            },
        'subject_name' : 'YH02VT',
        'output_filename' : 'YH02VT_PPC_passive.html'
        }
    session_config_list_YH03VT = {
        'list_config': [
            session_config_YH03VT,
            ],
        'label_names' : {
            '-1':'Exc',
            '1':'Inh_VIP',
            '2':'Inh_SST',
            },
        'subject_name' : 'YH03VT',
        'output_filename' : 'YH03VT_PPC_passive.html'
        }
    session_config_list_YH07VT = {
        'list_config': [
            session_config_YH07VT,
            ],
        'label_names' : {
            '-1':'Exc',
            '1':'Inh_VIP',
            '2':'Inh_SST',
            },
        'subject_name' : 'YH07VT',
        'output_filename' : 'YH07VT_PPC_passive.html'
        }
    session_config_list_all = {
        'list_config': [
            session_config_YH01VT,
            session_config_YH02VT,
            session_config_YH03VT,
            ],
        'label_names' : {
            '-1':'Exc',
            '1':'Inh_VIP',
            '2':'Inh_SST',
            },
        'subject_name' : 'all',
        'output_filename' : 'all_PPC_passive.html'
        }

#%% start processing

    run(session_config_list_YH01VT)
    run(session_config_list_YH02VT)
    run(session_config_list_YH03VT)
    run(session_config_list_all)
    
    '''
    
    session_config_test = {
        'list_session_name' : {
            'FN14_PPC_20250102_3331Random_test' : 'random',
            'FN14_PPC_20250102_1451ShortLong_test' : 'short_long',
            'FN14_PPC_20250102_4131FixJitterOdd_test' : 'fix_jitter_odd',
            },
        'session_folder' : 'FN14',
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
        'subject_name' : 'FN14',
        'output_filename' : 'FN14_PPC_passive.html'
        }
    run(session_config_list_test)
    
    session_config_test = {
        'list_session_name' : {
            'VTYH01_PPC_20250108_3331Random' : 'random',
            'VTYH01_PPC_20250116_1451ShortLong' : 'short_long',
            'VTYH01_PPC_20250123_4131FixJitterOdd' : 'fix_jitter_odd',
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
        'subject_name' : 'test',
        'output_filename' : 'test_PPC_passive.html'
        }
    run(session_config_list_test)
    
    session_config_list = combine_session_config_list(session_config_list_test)
    list_ops = read_ops(session_config_list['list_session_data_path'])
    [list_labels, list_masks, list_vol, list_dff,
     list_neural_trials, list_move_offset, list_significance
     ] = read_all(session_config_list)
    '''
