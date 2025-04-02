#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import gc
import os

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
    smooth = False

    print('===============================================')
    print('============ Start Processing Data ============')
    print('===============================================')
    if not os.path.exists(os.path.join('results', session_config_list['subject_name']+'_temp')):
        os.makedirs(os.path.join('results', session_config_list['subject_name']+'_temp'))
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
    #fn1 = visualization1_FieldOfView.run(session_config_list, smooth)
    fn1 = []

    print('===============================================')
    print('========= plotting 3331Random results =========')
    print('===============================================')
    fn2 = visualization2_3331Random.run(session_config_list, smooth)
    
    print('===============================================')
    print('======= plotting 1451ShortLong results ========')
    print('===============================================')
    fn3 = visualization3_1451ShortLong.run(session_config_list, smooth)

    print('===============================================')
    print('====== plotting 4131FixJitterOdd results ======')
    print('===============================================')
    fn4 = visualization4_4131FixJitterOdd.run(session_config_list, smooth)

    print('===============================================')
    print('============ saving session report ============')
    print('===============================================')
    print('Saving results')
    pack_webpage_main.run(session_config_list, fn1, fn2, fn3, fn4)
    for i in range(len(list_ops)):
        print('Cleaning memory mapping files for {}'.format(
            list(session_config_list['list_session_name'].keys())[i]))
        clean_memap_path(list_ops[i])
    print('Processing completed for all sessions')
    for n in session_config_list['list_session_name']:
        print(n)
    print('File saved as '+os.path.join('results', session_config_list['output_filename']))
    del list_ops
    gc.collect()


if __name__ == "__main__":

    from session_configs import session_config_list_YH01VT
    from session_configs import session_config_list_YH02VT
    from session_configs import session_config_list_YH03VT
    from session_configs import session_config_list_YH14SC
    from session_configs import session_config_list_YH17VT
    from session_configs import session_config_list_YH18VT
    from session_configs import session_config_list_YH19VT
    from session_configs import session_config_list_YH20SC
    from session_configs import session_config_list_PPC
    from session_configs import session_config_list_V1

    run(session_config_list_YH01VT)
    run(session_config_list_YH02VT)
    run(session_config_list_YH03VT)
    run(session_config_list_YH14SC)
    run(session_config_list_YH17VT)
    run(session_config_list_YH18VT)
    run(session_config_list_YH19VT)
    run(session_config_list_YH20SC)
    run(session_config_list_PPC)
    run(session_config_list_V1)

    '''
    session_config_test = {
        'list_session_name' : {
            #'VTYH02_PPC_20250108_3331Random' : 'random',
            #'VTYH02_PPC_20250109_3331Random' : 'random',
            #'VTYH02_PPC_20250111_3331Random' : 'random',
            #'VTYH02_PPC_20250131_4131FixJitterOdd' : 'fix_jitter_odd',
            #'VTYH02_PPC_20250202_4131FixJitterOdd' : 'fix_jitter_odd',
            #'VTYH02_PPC_20250203_4131FixJitterOdd' : 'fix_jitter_odd',
            'VTYH02_PPC_20250225_1451ShortLong' : 'short_long',
            'VTYH02_PPC_20250226_1415ShortLong' : 'short_long',
            'VTYH02_PPC_20250228_1451ShortLong' : 'short_long',
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
        'subject_name' : 'ppc_test',
        'output_filename' : 'test_passive.html'
        }
    #run(session_config_list_test)

    session_config_list = combine_session_config_list(session_config_list_test)
    list_ops = read_ops(session_config_list['list_session_data_path'])
    from modules.ReadResults import read_all
    [list_labels, list_masks, list_vol, list_dff,
     list_neural_trials, list_move_offset, list_significance
     ] = read_all(session_config_list, smooth=False)

    label_names = {'-1':'Exc', '1':'Inh_VIP', '2':'Inh_SST'}
    cate = [-1,1,2]
    roi_id = None
    norm_mode='none'
    import matplotlib.pyplot as plt
    cluster_cmap = plt.cm.hsv
    '''
