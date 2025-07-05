import os
from Modules.reader import read_ops
from Modules.reader import read_masks
from Modules.reader import read_raw_voltages
from Modules.reader import read_neural_trials
from Modules.reader import read_trial_label
from Modules.trialize import trialize
from Modules.reader import clean_memap_path
from Modules.clustering_neurons import clustering_GMM
from Modules.decoding import decoder_decision
from Plot.plot_FOV import create_superimposed_mask_images
from Plot.plot_FOV import plot_fov_summary
from Plot.plot_licking_neural_response import plot_licking_neural_response
from Plot.plot_events_neural_response import plot_events_neural_response


list_session_data_path = [
    'F:\\Single_Interval_discrimination\\Data_2p\\YH24LG_CRBL_simplex_20250601_2afc-392',
    'F:\\Single_Interval_discrimination\\Data_2p\\YH24LG_CRBL_simplex_20250530_2afc-389',
    # 'F:\\Single_Interval_discrimination\\Data_2p\\YH24LG_CRBL_simplex_20250529_2afc-379',
    
]

list_ops = read_ops(list_session_data_path)

list_labels = []
list_neural_trials = []

for i, session_path in enumerate(list_session_data_path):
    
    print('####################################################')
    print(f'Processing session: {session_path}')
    print('####################################################')

    # label -1 --> excitatory neurons, 1 --> inhibitory neurons
    labels, masks, mean_func, max_func, mean_anat, masks_anat = read_masks(list_ops[i])

    # plotting FOV
    mean_fun_channel, max_fun_channel, superimpose_mask_func, superimpose_mask_anat = create_superimposed_mask_images(
        mean_func, max_func, masks, labels, mean_anat
    )

    print('Plotting FOV')
    plot_fov_summary(
        mean_fun_channel, max_fun_channel, masks, superimpose_mask_func,
        mean_anat, superimpose_mask_anat, save_path=session_path
    )

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
    list_neural_trials.append(neural_trials)

    # single session plotting
    print('Plotting licking response')
    plot_licking_neural_response(neural_trials, labels, save_path=session_path, pooling=False)
    print('Plotting events response')
    plot_events_neural_response(neural_trials, labels, save_path=session_path, pooling=False)

    # Single session clustering
    # NOTE: uncomment if you want to have clustering for each session
    # print('Clustering neurons using GMM')
    # clustering_GMM(neural_trials, labels, save_path = session_path, pooling = False)  


if len(list_session_data_path) == 1:
    print('Only one session data provided, skipping summary figures and clustering with pooled sessions.')
    exit()

##### Summary figures
print('####################################################')
print('summary analysis for all sessions')
print('####################################################')
summay_path = 'F:\\Single_Interval_discrimination\\Data_2p\\Summary'
if not os.path.exists(summay_path):
    os.makedirs(summay_path)

print('Plotting summary licking response')
plot_licking_neural_response(list_neural_trials, list_labels, save_path=summay_path, pooling=True)
print('Plotting summary events response')
plot_events_neural_response(list_neural_trials, list_labels, save_path=summay_path, pooling=True)
print('Clustering neurons using GMM with pooled sessions')
clustering_GMM(list_neural_trials, list_labels, save_path=summay_path, pooling=True)
print('decoding neural data using SVM with pooled sessions')
decoder_decision(list_neural_trials, list_labels, l_frames = 60, r_frames = 120, save_path=summay_path, pooling=True)
