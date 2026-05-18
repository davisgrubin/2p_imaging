import os
from Modules.reader import read_ops
from Modules.reader import read_masks
from Modules.reader import read_raw_voltages
from Modules.reader import read_neural_trials
from Modules.reader import read_trial_label
from Modules.trialize import trialize
from Modules.reader import clean_memap_path
from Modules.clustering_neurons import clustering_GMM
# from Modules.decoding_four import decoder_decision
from Modules.decoding_multiple_settings import decoder_decision
from Plot.plot_FOV import create_superimposed_mask_images
from Plot.plot_FOV import plot_fov_summary
from Plot.plot_FOV import plot_cluster_FOV
from Plot.plot_licking_neural_response import plot_licking_neural_response
from Plot.plot_events_neural_response import plot_events_neural_response
from Plot.plot_epoch_response import plot_events_epoch_neural_response
from Plot.plot_licking import main_plot_licking_patterns

import numpy as np

list_session_data_path = [
    "F:\Single_Interval_discrimination\Data_2p\YH24LG_CRBL_lobulev_20250709_2afc-545",
    "F:\Single_Interval_discrimination\Data_2p\YH24LG_CRBL_lobulev_20250708_2afc-543"
]

list_ops = read_ops(list_session_data_path)

list_labels = []
list_trial_labels = []
list_neural_trials = []

list_masks = []
list_mean_func = []
list_max_func = []
list_mean_anat = []
list_masks_anat = []

for i, session_path in enumerate(list_session_data_path):
    
    print('####################################################')
    print(f'Processing session: {session_path}')
    print('####################################################')

    # label -1 --> excitatory neurons, 1 --> inhibitory neurons
    labels, masks, mean_func, max_func, mean_anat, masks_anat = read_masks(list_ops[i])

    list_masks.append(masks)
    list_mean_func.append(mean_func)
    list_max_func.append(max_func)
    list_mean_anat.append(mean_anat)
    list_masks_anat.append(masks_anat)

    # plotting FOV
    # mean_fun_channel, max_fun_channel, superimpose_mask_func, superimpose_mask_anat = create_superimposed_mask_images(
    #     mean_func, max_func, masks, labels, mean_anat
    # )

    # print('Plotting FOV')
    # plot_fov_summary(
    #     mean_fun_channel, max_fun_channel, masks, superimpose_mask_func,
    #     mean_anat, superimpose_mask_anat, save_path=session_path
    # )

    # Trialize and save csv (events timestamps) in each session data path, also save the processed voltage data in neural_trials.h5
    print('Trializing data and saving in .h5 and .csv files')
    trialize(list_ops[i])

    # Reading the saved trialized data for upcoming analysis
    neural_trials = read_neural_trials(list_ops[i], 1)  # 0 for not smoothing dff and 1 for smoothing dff
    trial_labels = read_trial_label(list_ops[i])
    # free up memory 
    clean_memap_path(list_ops[i])
    # appnending the labels and neural trials to the lists
    list_labels.append(labels)
    list_trial_labels.append(trial_labels)
    list_neural_trials.append(neural_trials)

    # single session plotting
    # print('Plotting licking response')
    # plot_licking_neural_response(neural_trials, labels, save_path=session_path, pooling=False)
    # print('Plotting events response')
    # plot_events_neural_response(neural_trials, labels, save_path=session_path, pooling=False)
    # print('Plotting epoch events response')
    # plot_events_epoch_neural_response(neural_trials, labels, save_path=session_path, pooling=False)
    print('Plotting Licking Patterns')
    main_plot_licking_patterns(list_trial_labels, save_path=session_path)


    # Single session clustering and decoding
    # NOTE: uncomment if you want to have clustering for each session
    # print('Clustering neurons using GMM')
    # clustering_labels = clustering_GMM(neural_trials, labels, save_path = session_path, pooling = False)  
    # plot_cluster_FOV(list_session_data_path, list_labels, list_masks, list_mean_func, list_max_func, list_mean_anat, list_masks_anat, clustering_labels)
    # print('Decoding neural data using SVM')
    # decoder_decision(neural_trials, labels, l_frames=60, r_frames=120, decoding = 'epoch', save_path=session_path, pooling=False)
    # decoder_decision(neural_trials, labels, l_frames = 60, r_frames = 120, decoding = 'trial_type', save_path=session_path, pooling=False)


if len(list_session_data_path) == 1:
    print('Only one session data provided, skipping summary figures and clustering with pooled sessions.')
    exit()

##### Summary figures
print('####################################################')
print('summary analysis for all sessions')
print('####################################################')
summay_path = 'F:\Single_Interval_discrimination\Data_2p\Summary'
if not os.path.exists(summay_path):
    os.makedirs(summay_path)

# print('Plotting summary licking response')
# plot_licking_neural_response(list_neural_trials, list_labels, save_path=summay_path, pooling=True)
# print('Plotting summary events response')
# plot_events_neural_response(list_neural_trials, list_labels, save_path=summay_path, pooling=True)
# print('Plotting summary epoch events response')
# plot_events_epoch_neural_response(list_neural_trials, list_labels, save_path=summay_path, pooling=True)
print('Plotting summary Licking Patterns')
main_plot_licking_patterns(list_trial_labels, save_path=summay_path)
# print('Clustering neurons using GMM with pooled sessions')
# clustering_labels = clustering_GMM(list_neural_trials, list_labels, save_path=summay_path, pooling=True)
# # Plotting clusters in FOV
# plot_cluster_FOV(list_session_data_path, list_labels, list_masks, list_mean_func, list_max_func, list_mean_anat, list_masks_anat, clustering_labels)

# print('decoding neural data using SVM with pooled sessions')
# decoder_decision(list_neural_trials, list_labels, l_frames=60, r_frames=120, save_path=summay_path, pooling=True)
# decoder_decision(list_neural_trials, list_labels, decoding = 'rare vs common',  indice = 0, l_frames = 60, r_frames = 120, save_path=summay_path, pooling=True)
# decoder_decision(list_neural_trials, list_labels, l_frames = 60, r_frames = 120, decoding = 'trial_type',save_path=summay_path, pooling=True)