#!/usr/bin/env python3

import sys
import warnings
warnings.filterwarnings('ignore')

import gc
import os
import argparse
from datetime import datetime

from modules import Trialization
from modules.ReadResults import read_ops

input_folder = 'C:\\Users\\saminnaji3\\Downloads\\passive\\' 
def combine_session_config_list(session_config_list):
    list_session_data_path = []
    for sc in session_config_list['list_config']:
        list_session_data_path += [
            os.path.join(input_folder, sc['session_folder'], n)
            for n in sc['list_session_name'].keys()]
    list_session_name = [sc['list_session_name'] for sc in session_config_list['list_config']]
    list_session_name = {k: v for d in list_session_name for k, v in d.items()}
    session_config_list['list_session_name'] = list_session_name
    session_config_list['list_session_data_path'] = list_session_data_path
    return session_config_list

# import visualization1_FieldOfView
# import visualization2_3331Random
# import visualization3_1451ShortLong
import visualization4_4131FixJitterOdd
# import visualization5_3331RandomExtended
from webpage import pack_webpage_main


def run(session_config_list, cate_list):
    smooth = False

    print('===============================================')
    print('============ Start Processing Data ============')
    print('===============================================')
    if not os.path.exists(os.path.join(input_folder, 'temp_'+session_config_list['subject_name'])):
        os.makedirs(os.path.join(input_folder, 'temp_'+session_config_list['subject_name']))
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
        #Trialization.run(list_ops[i], aud = 0)
        #Trialization.add_spikes(list_ops[i])

    print('===============================================')
    print('======== plotting representative masks ========')
    print('===============================================')
    #fn1 = visualization1_FieldOfView.run(session_config_list, smooth, cate_list)
    fn1 = []

    print('===============================================')
    print('========= plotting 3331Random results =========')
    print('===============================================')
    #fn2 = visualization2_3331Random.run(session_config_list, smooth, cate_list)
    fn2 = []

    print('===============================================')
    print('======= plotting 1451ShortLong results ========')
    print('===============================================')
    #fn3 = visualization3_1451ShortLong.run(session_config_list, smooth, cate_list)
    fn3 = []

    print('===============================================')
    print('====== plotting 4131FixJitterOdd results ======')
    print('===============================================')
    fn4 = visualization4_4131FixJitterOdd.run(session_config_list, smooth, cate_list)
    #fn4 = []
    
    print('===============================================')
    print('===== plotting 3331RandomExtended results =====')
    print('===============================================')
    #fn5 = visualization5_3331RandomExtended.run(session_config_list, smooth, cate_list)
    fn5 = []

    print('===============================================')
    print('============ saving session report ============')
    print('===============================================')
    print('Saving results')
    pack_webpage_main.run(
        session_config_list,
        [fn1, fn2, fn3, fn4, fn5],
        ['Field of View', 'the Random session', 'the Short-Long session', 'the Fix-Jitter-Oddball session', 'the Extended Random session'],
        ['random', 'fix_jitter_odd', 'short_long'])
    for i in range(len(list_ops)):
        print('Cleaning memory mapping files for {}'.format(
            list(session_config_list['list_session_name'].keys())[i]))
    print('Processing completed for all sessions')
    for n in session_config_list['list_session_name']:
        print(n)
    print('File saved as '+os.path.join('results', session_config_list['output_filename'] + '.html'))
    print('Finished at '+datetime.now().strftime("%Y.%m.%d %H:%M:%S"))
    del list_ops
    gc.collect()


if __name__ == "__main__":
    COMMANDLINE_MODE = 0
    cate_list = [[-1,1,2]]
    from session_configs import all_config_list
    
    if COMMANDLINE_MODE:
        parser = argparse.ArgumentParser(description='Experiments can go shit but Yicong will love you forever!')
        parser.add_argument('--config_list', required=True, type=str, help='Whether run denoising.')
        args = parser.parse_args()
        for subject, session_config_list in zip(
                ['SA15_LG', 'SA12_LG', 'SA13_LG', 'SA14_LG',
                 'AUD', 'VIS'],
                all_config_list
            ):
            if subject in args.config_list:
                run(session_config_list, cate_list)

    else:
        session_config_SA12 = {
            'list_session_name' : {
                # 'SA12_20250930' : 'fix_jitter_odd',
                'SA12_20251001' : 'fix_jitter_odd', # GOOD
                'SA12_20251002' : 'fix_jitter_odd', # GOOD
                # 'SA12_20251003' : 'fix_jitter_odd', # GOOD BUT LABELING PROBLEM
                # 'SA12_20251006' : 'fix_jitter_odd', # GOOD but sth wrong with the labaling
                'SA12_20251007' : 'fix_jitter_odd', # good
                'SA12_20251008' : 'fix_jitter_odd', # good
                # 'SA12_20251009' : 'fix_jitter_odd', # NOT GOOD SGNAL
                },
            'session_folder' : 'SA12_LG',
            'sig_tag' : 'all',
            'force_label' : None,
            }
        session_config_SA13 = {
            'list_session_name' : {
                'SA13_20250930' : 'fix_jitter_odd', # VERY GOOD
                'SA13_20251001' : 'fix_jitter_odd', # GOOD
                'SA13_20251002' : 'fix_jitter_odd', # VERY GOOD
                'SA13_20251003' : 'fix_jitter_odd', # VERY GOOD
                'SA13_20251006' : 'fix_jitter_odd', # VERY GOOD
                'SA13_20251007' : 'fix_jitter_odd', # GOOD
                # 'SA13_20251008' : 'fix_jitter_odd', # GOOD but sth wrong with the labaling
                'SA13_20251009' : 'fix_jitter_odd', # GOOD
                },
            'session_folder' : 'SA13_LG',
            'sig_tag' : 'all',
            'force_label' : None,
            }
        session_config_SA14 = {
            'list_session_name' : {
                # 'SA14_20251001' : 'fix_jitter_odd',
                'SA14_20251002' : 'fix_jitter_odd',
                'SA14_20251003' : 'fix_jitter_odd',
                # 'SA14_20251006' : 'fix_jitter_odd',
                'SA14_20251007' : 'fix_jitter_odd',
                'SA14_20251008' : 'fix_jitter_odd',
                'SA14_20251009' : 'fix_jitter_odd',
                # 'SA14_20251010' : 'fix_jitter_odd',
                },
            'session_folder' : 'SA14_LG',
            'sig_tag' : 'all',
            'force_label' : None,
            }
        session_config_SA15 = {
            'list_session_name' : {
                'SA15_20251002' : 'fix_jitter_odd',
                'SA15_20251003' : 'fix_jitter_odd',
                # 'SA15_20251006' : 'fix_jitter_odd',
                # 'SA15_20251007' : 'fix_jitter_odd',
                # 'SA15_20251008' : 'fix_jitter_odd',
                'SA15_20251009' : 'fix_jitter_odd',
                # 'SA15_20251010' : 'fix_jitter_odd',
                },
            'session_folder' : 'SA15_LG',
            'sig_tag' : 'all',
            'force_label' : None,
            }
        session_config_SA21 = {
            'list_session_name' : {
                'SA21_20260217' : 'fix_jitter_odd',
                'SA21_20260219' : 'fix_jitter_odd',
                'SA21_20260220' : 'fix_jitter_odd',
                'SA21_20260226' : 'fix_jitter_odd',
                'SA21_20260227' : 'fix_jitter_odd',
                'SA21_20260228' : 'fix_jitter_odd',
                'SA21_20260301' : 'fix_jitter_odd',
                },
            'session_folder' : 'SA21_LG',
            'sig_tag' : 'all',
            'force_label' : None,
            }
        session_config_list_test = {
            'list_session_name' : {
                'SA21_20260302' : 'fix_jitter_odd',
                },
            'session_folder' : 'SA21_LG',
            'sig_tag' : 'all',
            'force_label' : None,
            }
        
        session_config_list_test = {
            'list_config': [
                #session_config_SA15,
                #session_config_SA14,
                #session_config_SA21,
                #session_config_SA13,
                #session_config_SA12,
                session_config_list_test
                ],
            'label_names' : {
                '-1':'Exc',
                '1':'Inh_VIP',
                '2':'Inh_SST',
                },
            'subject_name' : 'SA21',
            'output_filename' : 'SA21_TEST'
            }

        run(session_config_list_test, cate_list)
        
        # import matplotlib.pyplot as plt
        # session_config_list = combine_session_config_list(session_config_list_test)
        # list_ops = read_ops(session_config_list['list_session_data_path'])
        # ops = list_ops[0]

        # from modules.ReadResults import read_all
        # [list_labels, list_masks,
        #  list_neural_trials, list_move_offset
        #  ] = read_all(session_config_list, smooth=False)
        # neural_trials = list_neural_trials[0]
        # dff = neural_trials['dff']
        # label_names = {'-1':'Exc', '1':'Inh_VIP', '2':'Inh_SST'}
        # cate = [-1,1,2]
        # roi_id = None
        # norm_mode='none'
        # jitter_trial_mode='global'
        # scaled=True
       
        # cluster_cmap = plt.cm.hsv
        # standard = 1
        # oddball = 1
        # block = 0
        # mode = 'post'
        # temp_folder = 'temp_'+session_config_list['subject_name']
        # if not os.path.exists(os.path.join('results', 'temp_'+session_config_list['subject_name'])):
        #     os.makedirs(os.path.join('results', 'temp_'+session_config_list['subject_name']))

        
