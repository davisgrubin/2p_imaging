#!/usr/bin/env python3

#%% YH21SC

session_config_YH24LG = {
    'list_session_name' : {
        'YH24LG_CRBL_crux1_20250425_2afc' : 'single',
        'YH24LG_CRBL_crux1_20250426_2afc' : 'single',
        'YH24LG_CRBL_crux1_20250428_2afc' : 'single',
        'YH24LG_CRBL_crux1_20250429_2afc' : 'single',
        },
    'session_folder' : 'YH24LG',
    'sig_tag' : 'all',
    'force_label' : -1,
    }

#%% crbl list configs

# YH01VT.
session_config_list_YH24LG = {
    'list_config': [
        session_config_YH24LG,
        ],
    'label_names' : {
        '-1':'crux1',
        },
    'subject_name' : 'YH24LG',
    'output_filename' : 'YH24LG_crbl_2afc_single.html'
    }
