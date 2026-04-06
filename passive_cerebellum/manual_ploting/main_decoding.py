# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:27:46 2026

@author: saminnaji3
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings

# Import your custom modules
from Modules import ReadResults
from Modules import main_plots

warnings.filterwarnings("ignore")

def decode_sequence_num(list_neural_data, list_labels, target_cluster=0, 
                        pre_stim=15, post_stim=30, n_shuffles=50, jitter_val=None):
    """
    Decodes stimulus sequence number (0-3) by calculating accuracy 
    per session and averaging the results.
    """
    session_accuracies = []
    session_shuffles = []

    for sess_idx, neural_data in enumerate(list_neural_data):
        X_sess = []
        y_sess = []
        
        stim_labels = neural_data['stim_labels']
        current_labels = list_labels[sess_idx]
        
        # Filter neurons by cluster
        if target_cluster == -1:
            neuron_idx = np.arange(neural_data['dff'].shape[0])
        else:
            neuron_idx = np.where(current_labels == target_cluster)[0]
            
        if len(neuron_idx) == 0: continue

        # Filter stims by jitter and sequence
        mask = (stim_labels[:, 8] >= 0) & (stim_labels[:, 8] <= 3)
        if jitter_val is not None:
            mask &= (stim_labels[:, 4] == jitter_val)
        
        selected_stims = stim_labels[mask]
        
        for stim in selected_stims:
            onset_idx = np.searchsorted(neural_data['time'], stim[0])
            start, end = onset_idx - pre_stim, onset_idx + post_stim
            
            if start < 0 or end > neural_data['dff'].shape[1]: continue
                
            # Individual neuron features for THIS session
            trial_features = neural_data['dff'][neuron_idx, start:end].flatten()
            X_sess.append(trial_features)
            y_sess.append(int(stim[8]))

        # Skip session if it has too few trials for 5-fold CV
        if len(X_sess) < 15: continue

        X = np.array(X_sess)
        y = np.array(y_sess)
        scaler = StandardScaler()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 1. Real Accuracy for this session
        fold_accs = []
        for train_idx, test_idx in skf.split(X, y):
            X_train = scaler.fit_transform(X[train_idx])
            clf = SVC(kernel='linear').fit(X_train, y[train_idx])
            fold_accs.append(accuracy_score(y[test_idx], clf.predict(scaler.transform(X[test_idx]))))
        session_accuracies.append(np.mean(fold_accs))
        
        # 2. Optional: Shuffled Accuracy for this session
        if n_shuffles > 0:
            y_shuff = y.copy()
            shuff_results = []
            for _ in range(n_shuffles):
                np.random.shuffle(y_shuff)
                s_accs = []
                for train_idx, test_idx in skf.split(X, y_shuff):
                    X_tr = scaler.fit_transform(X[train_idx])
                    clf_s = SVC(kernel='linear').fit(X_tr, y_shuff[train_idx])
                    s_accs.append(accuracy_score(y_shuff[test_idx], clf_s.predict(scaler.transform(X[test_idx]))))
                shuff_results.append(np.mean(s_accs))
            session_shuffles.append(np.mean(shuff_results))

    if not session_accuracies: return None, None

    return np.mean(session_accuracies), (np.mean(session_shuffles) if n_shuffles > 0 else 0.25)


def decode_temporal_course(list_neural_data, list_labels, target_cluster=0, 
                           window_size=10, step_size=5, start_offset=-15, end_offset=45,
                           jitter_val=None):
    """
    Slides a window across the stimulus onset to see how decoding accuracy changes over time.
    """
    # Define the centers of our sliding windows (relative to stim onset at 0)
    time_points = np.arange(start_offset, end_offset, step_size)
    course_acc = []
    
    for tp in time_points:
        # Define window for this specific time step
        pre = -tp + (window_size // 2)
        post = tp + (window_size // 2)
        
        # Use your existing decoding logic for this specific slice
        acc, _ = decode_sequence_num(
            list_neural_data, list_labels, 
            target_cluster=target_cluster,
            pre_stim=int(pre), post_stim=int(post), 
            n_shuffles=0, # Skip shuffle during the loop for speed
            jitter_val=jitter_val
        )
        course_acc.append(acc if acc is not None else np.nan)
        
    return time_points, np.array(course_acc)

def plot_temporal_decoding(list_neural_data, labels_per_session, output_path, cluster_id=0):
    fig, ax = plt.subplots(figsize=(10, 6))
    fs = 30.0 # Your sampling rate
    
    colors_dict = {'Fix': '#0072B2', 'Jitter': '#CC79A7', 'All': 'black'}
    
    for mode, j_val in [('Fix', 0), ('Jitter', 1), ('All', None)]:
        print(f"Processing temporal course for: {mode}")
        t_points, acc_course = decode_temporal_course(
            list_neural_data, labels_per_session, 
            target_cluster=cluster_id, jitter_val=j_val
        )
        
        # Convert sample offsets to seconds
        t_seconds = t_points / fs
        ax.plot(t_seconds, acc_course, label=mode, color=colors_dict[mode], linewidth=2)

    # Styling
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Stim Onset')
    ax.axhline(0.25, color='red', linestyle=':', alpha=0.8, label='Chance')
    
    # Use your standard layout helper
    main_plots.lay_out_plot(
        ax, 
        x_label='Time from Stimulus (s)', 
        y_label='Decoding Accuracy',
        title=f'Temporal Sequence Decoding (Cluster {cluster_id})',
        legend=1
    )
    
    plt.savefig(os.path.join(output_path, f'temporal_decoding_cluster_{cluster_id}.pdf'))
    plt.show()

# Run the analysis


subject = 'SA14_LG'
subject_id = 'SA14_'
date = '20260331/'
output_dir_onedrive = os.path.join('/storage/project/r-fnajafi3-0/shared/2P_Imaging/', subject, 'all_figs', date)
output_dir_onedrive = 'C:/Users/saminnaji3/Downloads/passive'
output_dir_local = output_dir_onedrive
initial_path = '/storage/project/r-fnajafi3-0/shared/2P_Imaging'

initial_path = 'C:/Users/saminnaji3/Downloads/passive'
# sa16
data_dates = ['20251215', '20251226', '20251227', '20251228', '20251229', '20251230', '20260103', '20260105' , '20260106', '20260107']
# sa18
data_dates = ['20251226', '20251227', '20251228', '20251229', '20251230', '20260103', '20260104', '20260105' , '20260106', '20260107']
# SA20
data_dates = ['20260120', '20260121']
# YH30
data_dates = ['20260130']

data_dates = ['20260217', '20260219', '20260220', '20260226', '20260227', '20260228', '20260301']
data_dates = ['20260304', '20260305', '20260306', '20260308', '20260310', '20260311', '20260314', '20260315']
data_dates = [
    '20260217', '20260219', '20260220', '20260226', '20260227', '20260228', 
    '20260301', '20260304', '20260305', '20260306', '20260308', '20260310', '20260311', '20260314', '20260315',
    '20260316', '20260319', '20260320', '20260321',]

data_dates = ['20251002', '20251003', '20251007', '20251008', '20251009']

list_session_data_path = []
for date in data_dates:
    list_session_data_path.append(os.path.join(initial_path, subject, subject_id + date))
  

list_neural_data = []
labels_per_session = []
for session_data_path in list_session_data_path:
    ops = np.load(
        os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session_data_path)
    neural_data = ReadResults.read_neural_data(ops)
    # --- ISI CALCULATION LOGIC ---
    stim_labels = neural_data['stim_labels']
    onsets = stim_labels[:, 0]  # Column 0 is trigger time (ms or s)
    
    # Calculate differences between consecutive onsets
    intervals = np.diff(onsets)
    
    # 1. Preceding ISI: [NaN, onset1-onset0, onset2-onset1, ...]
    pre_isi = np.concatenate([[np.nan], intervals])
    
    # 2. Next ISI: [onset1-onset0, onset2-onset1, ..., NaN]
    next_isi = np.concatenate([intervals, [np.nan]])
    
    # Append these as new columns (horizontally stack)
    # This adds them to the end of the existing columns
    updated_labels = np.column_stack((stim_labels, pre_isi, next_isi))
    
    # Update the dictionary
    neural_data['stim_labels'] = updated_labels
    # -----------------------------
    list_neural_data.append(neural_data)
    labels = ReadResults.read_cluster_labels(ops)
    labels_per_session.append(labels)



plot_temporal_decoding(list_neural_data, labels_per_session, output_dir_onedrive, cluster_id=0)

# --- EXECUTION BLOCK ---
# Using the session lists you defined in your snippet
# --- CLUSTER ANALYSIS LOOP ---
# -1: all neurons, 0 & 1: specific clusters, 'all': iterate and do separately
clusters_to_run = [-1, 0, 1] 

for cluster_id in clusters_to_run:
    print(f"\n{'='*30}")
    print(f"STARTING ANALYSIS FOR CLUSTER: {cluster_id}")
    print(f"{'='*30}")
    
    # 1. Temporal Course Analysis
    # This will save a PDF: temporal_decoding_cluster_{cluster_id}.pdf
    plot_temporal_decoding(list_neural_data, labels_per_session, output_dir_onedrive, cluster_id=cluster_id)
    
    # 2. Bar Chart Summary Analysis
    results = {}
    for mode, j_val in [('Fix', 0), ('Jitter', 1), ('All', None)]:
        print(f"Running Summary Decoding for: {mode}...")
        acc, shuff = decode_sequence_num(
            list_neural_data, 
            labels_per_session, 
            target_cluster=cluster_id, 
            jitter_val=j_val
        )
        if acc is not None:
            results[mode] = {'Acc': acc, 'Shuff': shuff}

    # 3. Save Summary Bar Chart PDF
    if results:
        fig, ax = plt.subplots(figsize=(10, 6))
        modes = list(results.keys())
        x = np.arange(len(modes))
        
        ax.bar(x - 0.2, [results[m]['Acc'] for m in modes], 0.4, label='Decoder', color='#0072B2')
        ax.bar(x + 0.2, [results[m]['Shuff'] for m in modes], 0.4, label='Shuffle', color='gray', alpha=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.axhline(0.25, color='red', linestyle='--', label='Chance (1/4)')
        
        main_plots.lay_out_plot(
            ax, 
            x_label='Session Type', 
            y_label='Accuracy', 
            title=f'Summary Seq Decoding (Cluster {cluster_id})', 
            legend=1
        )
        
        summary_filename = f'sequence_decoding_summary_cluster_{cluster_id}.pdf'
        plt.savefig(os.path.join(output_dir_onedrive, summary_filename), bbox_inches='tight')
        plt.close(fig) # Close to free memory during loop
        print(f"Saved summary to: {summary_filename}")

print("\nAll cluster analyses complete.")