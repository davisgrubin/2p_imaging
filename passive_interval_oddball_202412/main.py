#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import os
import fitz

from modules import Trialization
from modules import StatTest
from modules.ReadResults import read_ops
from modules.ReadResults import read_all

def get_roi_sign(significance, roi_id):
    r = significance['r_standard'][roi_id] +\
        significance['r_change'][roi_id] +\
        significance['r_oddball'][roi_id]
    return r

import visualization1_masks
import visualization2_3331Random
import visualization3_1451ShortLong
import visualization4_4131FixJitterOdd

def run(session_config):

    print('===============================================')
    print('============ Start Processing Data ============')
    print('===============================================')
    list_session_data_path = [
        os.path.join('results', n)
        for n in session_config['list_session_name'].keys()]
    print('Processing {} sessions'.format(len(list_session_data_path)))
    for n in session_config['list_session_name'].keys():
        print(n)
    print('Reading ops.npy')
    list_ops = read_ops(list_session_data_path)

    print('===============================================')
    print('============= trials segmentation =============')
    print('===============================================')
    for i in range(len(list_ops)):
        print('Trializing {}'.format(
            list(session_config['list_session_name'].keys())[i]))
        Trialization.run(list_ops[i])

    print('===============================================')
    print('============== significance test ==============')
    print('===============================================')
    for i in range(len(list_ops)):
        print('Running significance test for {}'.format(
            list(session_config['list_session_name'].keys())[i]))
        StatTest.run(list_ops[i])

    print('===============================================')
    print('============ reading saved results ============')
    print('===============================================')
    [list_labels, list_masks, list_vol, list_dff,
     list_neural_trials, list_move_offset, list_significance
     ] = read_all(list_ops, session_config['sig_tag'], session_config['force_label'])
    print('Read {} session results'.format(len(list_ops)))

    print('===============================================')
    print('============= Start visualization =============')
    print('===============================================')
    session_report = fitz.open()
    print('Created canvas')

    print('===============================================')
    print('======== plotting representative masks ========')
    print('===============================================')
    visualization1_masks.run(
        session_config, session_report,
        list_labels, list_masks, list_vol, list_dff, list_move_offset)

    print('===============================================')
    print('========= plotting 3331Random results =========')
    print('===============================================')
    visualization2_3331Random.run(
        session_config, session_report,
        list_labels, list_vol, list_dff, list_neural_trials, list_significance)

    print('===============================================')
    print('======= plotting 1451ShortLong results ========')
    print('===============================================')
    visualization3_1451ShortLong.run(
        session_config, session_report,
        list_labels, list_vol, list_dff, list_neural_trials, list_significance)

    print('===============================================')
    print('====== plotting 4131FixJitterOdd results ======')
    print('===============================================')
    visualization4_4131FixJitterOdd.run(
        session_config, session_report,
        list_labels, list_vol, list_dff, list_neural_trials, list_significance)

    print('===============================================')
    print('============ saving session report ============')
    print('===============================================')
    session_report.save(os.path.join('results', session_config['output_filename']))
    session_report.close()
    for n in session_config['list_session_name'].keys():
        print(n)
    print('Processing completed')
    print('File saved as '+os.path.join('results', session_config['output_filename']))

if __name__ == "__main__":
    
    session_config_VTYH01 = {
        'list_session_name' : {
            'VTYH01_PPC_20250106_3331Random' : 'random',
            'VTYH01_PPC_20250107_3331Random' : 'random',
            'VTYH01_PPC_20250108_3331Random' : 'random',
            },
        'sig_tag' : 'all',
        'label_names' : {
            '-1':'excitatory',
            '1':'inhibitory'
            },
        'force_label' : None,
        'output_filename' : 'VTYH01_PPC_passive.pdf'
        }
    session_config_VTYH02 = {
        'list_session_name' : {
            'VTYH02_PPC_20250106_3331Random' : 'random',
            'VTYH02_PPC_20250107_3331Random' : 'random',
            'VTYH02_PPC_20250108_3331Random' : 'random',
            },
        'sig_tag' : 'all',
        'label_names' : {
            '-1':'excitatory',
            '1':'inhibitory'
            },
        'force_label' : None,
        'output_filename' : 'VTYH02_PPC_passive.pdf'
        }
    session_config_VTYH03 = {
        'list_session_name' : {
            'VTYH03_PPC_20250106_3331Random' : 'random',
            'VTYH03_PPC_20250107_3331Random' : 'random',
            'VTYH03_PPC_20250108_3331Random' : 'random',
            },
        'sig_tag' : 'all',
        'label_names' : {
            '-1':'excitatory',
            '1':'inhibitory'
            },
        'force_label' : None,
        'output_filename' : 'VTYH03_PPC_passive.pdf'
        }
    
    run(session_config_VTYH01)
    run(session_config_VTYH02)
    run(session_config_VTYH03)

    '''
    
    session_config_FN14 = {
        'list_session_name' : {
            'FN14_PPC_20250102_3331Random_test' : 'random',
            'FN14_PPC_20250102_1451ShortLong_test' : 'short_long',
            'FN14_PPC_20250102_4131FixJitterOdd_test' : 'fix_jitter_odd',
            },
        'sig_tag' : 'all',
        'label_names' : {
            '-1':'excitatory',
            '1':'inhibitory'
            },
        'force_label' : None,
        'output_filename' : 'FN14_PPC_passive.pdf'
        }
    run(session_config_FN14)
    
    session_config = session_config_VTYH03
    list_session_data_path = [
        os.path.join('results', n)
        for n in session_config['list_session_name'].keys()]
    list_ops = read_ops(list_session_data_path)
    [list_labels, list_masks, list_vol, list_dff,
     list_neural_trials, list_move_offset, list_significance
     ] = read_all(list_ops, session_config['sig_tag'], session_config['force_label'])
    
    '''
