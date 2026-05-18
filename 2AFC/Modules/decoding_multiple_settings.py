import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils import resample
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from Modules.Alignment import get_perception_response
import os


def neural_decoder_svm(data, labels, neu_time, trial_type, isi, label_names, shuffle = None, bin_data=False, bin_size=3, test_size=0.2, 
                      cv_folds= 5, C=0.1, undersample=True, random_state=42, save_path=None):
    """
    SVM decoder for 2AFC calcium imaging data with 4 labels.
    
    Parameters:
    -----------
    data : np.array
        Neural data with shape (n_trials, n_neurons, n_timepoints)
    labels : np.array
        Trial labels (0: short_left, 1: short_right, 2: long_left, 3: long_right)
    bin_data : bool
        Whether to bin the temporal data (default: False)
    bin_size : int
        Number of timepoints to bin together (default: 3)
    test_size : float
        Proportion of data for testing (default: 0.2)
    cv_folds : int
        Number of cross-validation folds (default: 5)
    C : float
        SVM regularization parameter (default: 1.0)
    undersample : bool
        Whether to randomly undersample majority classes (default: True)
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    results : dict
        Dictionary containing decoder results and metrics
    """
    
    np.random.seed(random_state)
    
    print("=== Neural Decoder Analysis ===")
    print(f"Original data shape: {data.shape}")
    
    # Step 1: Data preprocessing
    if bin_data:
        print(f"Binning data with bin size: {bin_size}")
        n_trials, n_neurons, n_timepoints = data.shape
        n_bins = n_timepoints // bin_size
        
        # Reshape and bin the data
        binned_data = data[:, :, :n_bins*bin_size].reshape(n_trials, n_neurons, n_bins, bin_size)
        data_processed = np.mean(binned_data, axis=3)  # Average within bins
        print(f"Binned data shape: {data_processed.shape}")
    else:
        print("Using unbinned data")
        data_processed = data.copy()
    
    # Step 2: Flatten the data (trials, neurons*timepoints)
    n_trials, n_neurons, n_timepoints = data_processed.shape
    X = data_processed.reshape(n_trials, n_neurons * n_timepoints)
    y = labels.copy()

    # Step 2.1: Handle NaN values and analyze label distribution
    # Identify non-NaN indices
    valid_indices = (~np.isnan(labels)) & (labels != -1)
    X = X[valid_indices]
    y = labels[valid_indices]
    
    print(f"Flattened feature matrix shape: {X.shape}")
    
    # Step 3: Analyze label distribution
    label_names = label_names #['Early', 'Late']
    label_counts = Counter(y)
    
    print("\n=== Label Distribution ===")
    for i, name in enumerate(label_names):
        count = label_counts.get(i, 0)
        print(f"{name}: {count} trials ({count/len(y)*100:.1f}%)")
    
    # Step 4: Handle class imbalance with undersampling
    if undersample and len(set(label_counts.values())) > 1:  # Only if imbalanced
        print("\n=== Undersampling Majority Classes ===")
        min_count = min(label_counts.values())
        print(f"Undersampling to {min_count} trials per class")
        
        # Combine data and labels for resampling
        indices_to_keep = []
        
        for label in range(2):
            label_indices = np.where(y == label)[0]
            if len(label_indices) > min_count:
                # Randomly sample min_count indices
                sampled_indices = np.random.choice(label_indices, min_count, replace=False)
                indices_to_keep.extend(sampled_indices)
            else:
                indices_to_keep.extend(label_indices)
        
        # Reorder and subset the data
        indices_to_keep = np.array(indices_to_keep)
        X = X[indices_to_keep]
        y = y[indices_to_keep]
        
        # Print new distribution
        new_label_counts = Counter(y)
        print("New label distribution:")
        for i, name in enumerate(label_names):
            count = new_label_counts.get(i, 0)
            print(f"{name}: {count} trials ({count/len(y)*100:.1f}%)")
    
    # Step 5: Train-test split with stratification
    print(f"\n=== Train-Test Split ({int((1-test_size)*100)}/{int(test_size*100)}) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} trials")
    print(f"Test set: {X_test.shape[0]} trials")
    
    # Step 6: Feature scaling
    print("\n=== Feature Scaling ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Applied z-score normalization")
    
    # Step 7: SVM with L2 regularization and balanced class weights
    print(f"\n=== SVM Training (C={C}) ===")
    svm_model = SVC(
        kernel='linear',
        C=C,  # L2 regularization (smaller C = more regularization)
        class_weight='balanced',  # Handle any remaining class imbalance
        random_state=random_state
    )
    
    # Step 8: Cross-validation on training set
    print(f"\n=== {cv_folds}-Fold Cross-Validation ===")
    cv_scores = cross_val_score(
        svm_model, X_train_scaled, y_train, 
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring='balanced_accuracy'
    )
    
    print(f"CV Balanced Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"CV Scores: {[f'{score:.3f}' for score in cv_scores]}")
    
    # Step 9: Train final model and evaluate on test set
    svm_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = svm_model.predict(X_train_scaled)
    y_test_pred = svm_model.predict(X_test_scaled)
    
    # Calculate metrics
    train_balanced_acc = balanced_accuracy_score(y_train, y_train_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    
    print(f"\n=== Final Results ===")
    print(f"Training Balanced Accuracy: {train_balanced_acc:.3f}")
    print(f"Test Balanced Accuracy: {test_balanced_acc:.3f}")
    
    # Step 10: Detailed classification report
    print(f"\n=== Classification Report (Test Set) ===")
    target_names = [f'{i}_{name}' for i, name in enumerate(label_names)]
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    # save text of the report
    if save_path is not None:
        report_dir = os.path.join(save_path, 'Figures')
        os.makedirs(report_dir, exist_ok=True)
        shuffle_str = f"_{shuffle}" if shuffle is not None else ""
        report_path = os.path.join(report_dir, f'decoder_classification_report{shuffle_str}.txt')
        with open(report_path, 'w') as f:
            f.write(classification_report(y_test, y_test_pred, target_names=target_names))
    
    # Step 11: Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    if save_path is not None:
        figures_dir = os.path.join(save_path, 'Figures')
        os.makedirs(figures_dir, exist_ok=True)
        # Add shuffle info to filename if present
        shuffle_str = f"_{shuffle}" if shuffle is not None else ""
        fig_path = os.path.join(figures_dir, f'Confusion Matrix{shuffle_str}_{label_names[0]}.pdf')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()
    
    # Step 12: Feature importance (SVM weights)
    feature_weights = svm_model.coef_
    print(f"\nSVM decision function weights shape: {feature_weights.shape}")
    
    # For multiclass SVM, we get one weight vector per class comparison
    # Average absolute weights across all comparisons
    avg_weights = np.mean(np.abs(feature_weights), axis=0)
    
    # Reshape back to (neurons, timepoints) for visualization
    if bin_data:
        weight_matrix = avg_weights.reshape(n_neurons, n_bins)
        time_label = f'Time Bins (bin_size={bin_size})'
    else:
        weight_matrix = avg_weights.reshape(n_neurons, n_timepoints)
        time_label = 'Time Points'
    
    
    # Plot heatmap with neu_time as x-axis labels (if shape matches)
    if weight_matrix.shape[1] == len(neu_time):
        # Create a figure with subplots: heatmap on top, distributions below
        fig = plt.figure(figsize=(10, 8))

        # Create custom grid layout to control colorbar placement
        gs = fig.add_gridspec(2, 2, height_ratios=[5, 1], width_ratios=[20, 1], 
                            hspace=0.1, wspace=0.05)

        ax_heatmap = fig.add_subplot(gs[0, 0])
        ax_cbar = fig.add_subplot(gs[0, 1])
        ax_dist = fig.add_subplot(gs[1, 0])

        # Plot heatmap
        im = ax_heatmap.imshow(weight_matrix, cmap='inferno', aspect='auto', 
                            extent=[neu_time[0], neu_time[-1], 0, weight_matrix.shape[0]])

        # Add colorbar in dedicated subplot
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('Average |Weight|', rotation=270, labelpad=20)

        # Set heatmap labels and ticks
        ax_heatmap.set_ylabel('Neuron/Channel')
        ax_heatmap.set_xticks(np.linspace(neu_time[0], neu_time[-1], 10))
        ax_heatmap.set_xticklabels(np.round(np.linspace(neu_time[0], neu_time[-1], 10), 1), 
                                rotation=45)

        # Add vertical line at time zero in heatmap
        ax_heatmap.axvline(x=0, color='white', linestyle='--', linewidth=2)
        ax_heatmap.set_title('SVM Feature Importance Heatmap')

        # Plot ISI distributions without artificial scaling
        sns.kdeplot(x=isi[trial_type == 0] + 200, label='Short', color='blue', 
                    ax=ax_dist, bw_adjust=1.5)
        sns.kdeplot(x=isi[trial_type == 1] + 200, label='long', color='red', 
                    ax=ax_dist, bw_adjust=1.5)

        # Customize distribution plot
        ax_dist.set_xlabel('Time')
        ax_dist.set_ylabel('Density')
        ax_dist.spines['top'].set_visible(False)
        ax_dist.spines['right'].set_visible(False)
        ax_dist.legend()

        # Match x-axis limits
        ax_dist.set_xlim(neu_time[0], neu_time[-1])

        # Remove x-axis labels from heatmap since they're shared
        ax_heatmap.set_xticklabels([])
        # plt.tight_layout()
    else:
        plt.figure(figsize=(12, 6))
        sns.heatmap(weight_matrix, cmap='inferno', cbar_kws={'label': 'Average |Weight|'})
        plt.title('SVM Feature Importance Heatmap')
        plt.xlabel(time_label)
        plt.ylabel('Neurons')
        plt.tight_layout()

    if save_path is not None:
        figures_dir = os.path.join(save_path, 'Figures')
        os.makedirs(figures_dir, exist_ok=True)
        # Add shuffle info to filename if present
        shuffle_str = f"_{shuffle}" if shuffle is not None else ""
        fig_path = os.path.join(figures_dir, f'SVM_Feature_Importance_Heatmap{shuffle_str}_{label_names[0]}.pdf')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
    plt.close()
    
    # Step 13: Compile results
    results = {
        'model': svm_model,
        'scaler': scaler,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_balanced_accuracy': train_balanced_acc,
        'test_balanced_accuracy': test_balanced_acc,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'confusion_matrix': cm,
        'feature_weights': feature_weights,
        'weight_matrix': weight_matrix,
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'final_label_counts': Counter(y),
        'parameters': {
            'bin_data': bin_data,
            'bin_size': bin_size,
            'C': C,
            'cv_folds': cv_folds,
            'test_size': test_size,
            'undersample': undersample,
            'random_state': random_state
        }
    }
    
    return results

def pool_session_data(neural_trials_list, labels_list, state, l_frames, r_frames, indices):
    """
    Pool data from multiple sessions.
    
    Returns:
    tuple: Pooled neu_seq (n_trials_total, n_neurons_total, time), neu_time, trial_type, isi, decision, labels
    """
    neu_seqs = []
    trial_types = []
    block_types = []
    isis = []
    decisions = []
    all_labels = []
    all_outcomes = []

    neuron_counts = [labels.shape[0] for labels in labels_list]
    total_neurons = sum(neuron_counts)

    neuron_offset = 0  # for assigning each session's neurons into the correct padded space

    for i, (neural_trials, session_labels) in enumerate(zip(neural_trials_list, labels_list)):
        neu_seq, neu_time, stim_seq, stim_value, stim_time, led_value, trial_type, block_type, isi, decision, outcome = get_perception_response(
            neural_trials, state, l_frames, r_frames, indices=indices)

        n_trials, n_neurons, n_time = neu_seq.shape
        padded_neu_seq = np.zeros((n_trials, total_neurons, n_time))

        # Place this session’s neurons in the correct slice
        padded_neu_seq[:, neuron_offset:neuron_offset + n_neurons, :] = neu_seq
        neu_seqs.append(padded_neu_seq)

        trial_types.append(trial_type)
        block_types.append(block_type)
        isis.append(isi)
        decisions.append(decision)
        all_labels.append(session_labels)
        all_outcomes.append(outcome)

        neuron_offset += n_neurons

    # Concatenate along trials
    pooled_neu_seq = np.concatenate(neu_seqs, axis=0)  # (total_trials, total_neurons, time)
    pooled_trial_type = np.concatenate(trial_types, axis=0)
    pooled_block_type = np.concatenate(block_types, axis=0)
    pooled_isi = np.concatenate(isis, axis=0)
    pooled_decision = np.concatenate(decisions, axis=0)
    pooled_labels = np.concatenate(all_labels, axis=0)
    pooled_outcomes = np.concatenate(all_outcomes, axis=0)

    return pooled_neu_seq, neu_time, pooled_trial_type, pooled_block_type, pooled_isi, pooled_decision, pooled_labels, pooled_outcomes

def temporal_decoding_accuracy(data, labels, neu_time, trial_type, isi, window_size=5, 
                               step_size=2, n_iterations=10, test_size=0.2, C=0.1, 
                               undersample=True, random_state=42, save_path=None):
    """
    Analyze decoding accuracy as a function of time using sliding windows.
    
    Parameters:
    -----------
    data : np.array
        Neural data with shape (n_trials, n_neurons, n_timepoints)
    labels : np.array
        Trial labels (0: short_left, 1: short_right, 2: long_left, 3: long_right)
    neu_time : np.array
        Time points corresponding to neural data
    window_size : int
        Size of sliding window in time points (default: 10)
    step_size : int
        Step size for sliding window (default: 5)
    n_iterations : int
        Number of iterations for bootstrap resampling (default: 10)
    test_size : float
        Proportion of data for testing (default: 0.2)
    C : float
        SVM regularization parameter (default: 0.1)
    undersample : bool
        Whether to randomly undersample majority classes (default: True)
    random_state : int
        Random seed for reproducibility (default: 42)
    save_path : str
        Path to save results (default: None)
    
    Returns:
    --------
    results : dict
        Dictionary containing temporal decoding results
    """
    
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    from collections import Counter
    import numpy as np
    
    print("=== Temporal Decoding Accuracy Analysis ===")
    print(f"Data shape: {data.shape}")
    print(f"Window size: {window_size}, Step size: {step_size}")
    print(f"Number of bootstrap iterations: {n_iterations}")
    
    n_trials, n_neurons, n_timepoints = data.shape
    
    # Generate sliding window positions
    window_starts = np.arange(0, n_timepoints - window_size + 1, step_size)
    window_centers = window_starts + window_size // 2
    window_times = neu_time[window_centers]
    
    print(f"Number of time windows: {len(window_starts)}")
    
    # Initialize results storage
    accuracies_normal = []
    accuracies_shuffled = []
    
    # Process each time window
    for i, start_idx in enumerate(window_starts):
        end_idx = start_idx + window_size
        window_time = neu_time[window_centers[i]]
        
        print(f"Processing window {i+1}/{len(window_starts)}: "
              f"t={window_time:.1f}s (indices {start_idx}-{end_idx})")
        
        # Extract data for current window
        window_data = data[:, :, start_idx:end_idx]
        
        # Flatten window data (trials, neurons*timepoints_in_window)
        X_window = window_data.reshape(n_trials, n_neurons * window_size)
        
        # Bootstrap iterations for this window
        window_acc_normal = []
        window_acc_shuffled = []
        
        for iteration in range(n_iterations):
            # Set seed for reproducibility within each iteration
            iteration_seed = random_state + iteration
            np.random.seed(iteration_seed)
            
            # Handle class imbalance with undersampling
            y_balanced = labels.copy()
            X_balanced = X_window.copy()

            # Remove NaN labels
            valid_indices = ~np.isnan(y_balanced)
            X_balanced = X_balanced[valid_indices]
            y_balanced = y_balanced[valid_indices]
            
            if undersample:
                label_counts = Counter(y_balanced)
                if len(set(label_counts.values())) > 1:  # Only if imbalanced
                    min_count = min(label_counts.values())
                    indices_to_keep = []
                    
                    for label in range(4):
                        label_indices = np.where(y_balanced == label)[0]
                        if len(label_indices) > min_count:
                            sampled_indices = np.random.choice(label_indices, min_count, replace=False)
                            indices_to_keep.extend(sampled_indices)
                        else:
                            indices_to_keep.extend(label_indices)
                    
                    indices_to_keep = np.array(indices_to_keep)
                    X_balanced = X_balanced[indices_to_keep]
                    y_balanced = y_balanced[indices_to_keep]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=test_size, 
                stratify=y_balanced, random_state=iteration_seed
            )
            
            # Feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train SVM - Normal labels
            svm_model = SVC(kernel='linear', C=C, class_weight='balanced', 
                          random_state=iteration_seed)
            svm_model.fit(X_train_scaled, y_train)
            y_pred_normal = svm_model.predict(X_test_scaled)
            acc_normal = balanced_accuracy_score(y_test, y_pred_normal)
            window_acc_normal.append(acc_normal)
            
            # Train SVM - Shuffled labels
            y_train_shuffled = np.random.permutation(y_train)
            svm_model_shuffled = SVC(kernel='linear', C=C, class_weight='balanced', 
                                   random_state=iteration_seed)
            svm_model_shuffled.fit(X_train_scaled, y_train_shuffled)
            y_pred_shuffled = svm_model_shuffled.predict(X_test_scaled)
            acc_shuffled = balanced_accuracy_score(y_test, y_pred_shuffled)
            window_acc_shuffled.append(acc_shuffled)
        
        # Store results for this window
        accuracies_normal.append(window_acc_normal)
        accuracies_shuffled.append(window_acc_shuffled)
    
    # Convert to numpy arrays for easier manipulation
    accuracies_normal = np.array(accuracies_normal)  # (n_windows, n_iterations)
    accuracies_shuffled = np.array(accuracies_shuffled)
    
    # Calculate statistics
    mean_acc_normal = np.mean(accuracies_normal, axis=1)
    sem_acc_normal = np.std(accuracies_normal, axis=1) / np.sqrt(n_iterations)
    mean_acc_shuffled = np.mean(accuracies_shuffled, axis=1)
    sem_acc_shuffled = np.std(accuracies_shuffled, axis=1) / np.sqrt(n_iterations)
    
    results = {
        'window_times': window_times,
        'mean_acc_normal': mean_acc_normal,
        'sem_acc_normal': sem_acc_normal,
        'mean_acc_shuffled': mean_acc_shuffled,
        'sem_acc_shuffled': sem_acc_shuffled,
        'all_acc_normal': accuracies_normal,
        'all_acc_shuffled': accuracies_shuffled,
        'parameters': {
            'window_size': window_size,
            'step_size': step_size,
            'n_iterations': n_iterations,
            'test_size': test_size,
            'C': C,
            'undersample': undersample,
            'random_state': random_state
        }
    }
    
    return results

def plot_temporal_decoding_accuracy(results, isi, trial_type, decoding, save_path=None):
    """
    Plot temporal decoding accuracy with error bars and ISI distributions.
    
    Parameters:
    -----------
    results : dict
        Results from temporal_decoding_accuracy function
    isi : np.array
        Inter-stimulus intervals
    trial_type : np.array
        Trial types (0: short, 1: long)
    save_path : str
        Path to save the figure (default: None)
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), 
                                   gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # Plot temporal decoding accuracy
    window_times = results['window_times']
    mean_acc_normal = results['mean_acc_normal']
    sem_acc_normal = results['sem_acc_normal']
    mean_acc_shuffled = results['mean_acc_shuffled']
    sem_acc_shuffled = results['sem_acc_shuffled']
    
    # Plot normal accuracy with error bars
    ax1.plot(window_times, mean_acc_normal, label='Normal', color='black', linewidth=2)
    ax1.fill_between(window_times, 
                     mean_acc_normal - sem_acc_normal,
                     mean_acc_normal + sem_acc_normal,
                     alpha=0.3, color='black')
    
    # Plot shuffled accuracy with error bars
    ax1.plot(window_times, mean_acc_shuffled, label='Shuffled', color='gray', 
             linestyle='--', linewidth=2)
    ax1.fill_between(window_times,
                     mean_acc_shuffled - sem_acc_shuffled,
                     mean_acc_shuffled + sem_acc_shuffled,
                     alpha=0.3, color='gray')
    
    # Add chance level line (25% for 4-class classification)
    ax1.axhline(y=0.50, color='red', linestyle=':', alpha=0.7, label='Chance (25%)')
    
    # Add vertical line at time zero
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Stim1 onset')
    
    ax1.set_ylabel('Decoding Accuracy')
    ax1.set_title('Temporal Decoding Accuracy: Normal vs Shuffled')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, 1)
    
    # Plot ISI distributions
    if isi is not None and trial_type is not None:
        sns.kdeplot(x=isi[trial_type == 0] + 200, label='Short', color='blue', 
                   ax=ax2, bw_adjust=1.5)
        sns.kdeplot(x=isi[trial_type == 1] + 200, label='Long', color='red', 
                   ax=ax2, bw_adjust=1.5)
        ax2.legend()
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('ISI Density')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path is not None:
        figures_dir = os.path.join(save_path, 'Figures')
        os.makedirs(figures_dir, exist_ok=True)
        fig_path = os.path.join(figures_dir, f'temporal_decoding_accuracy_{decoding}.pdf')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
    plt.close()

def plot_shuffle_vs_normal(results, results_shuffled, neu_time, isi, trial_type, save_path=None):
    """
    Plot comparison of average SVM feature weights (across neurons) for normal and shuffled labels,
    and show ISI distributions below. Add a vertical line at time zero.
    """

    # Average across neurons
    mean_weights = np.mean(results['weight_matrix'], axis=0)
    sem_weights = np.std(results['weight_matrix'], axis=0) / np.sqrt(results['weight_matrix'].shape[0])
    # For shuffled results
    mean_weights_shuffled = np.mean(results_shuffled['weight_matrix'], axis=0)
    sem_weights_shuffled = np.std(results_shuffled['weight_matrix'], axis=0) / np.sqrt(results_shuffled['weight_matrix'].shape[0])

    # Create figure with two subplots: weights and ISI distributions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

    # Plot average weights
    ax1.plot(neu_time, mean_weights, label='Normal', color='black')
    ax1.fill_between(neu_time, 
                     mean_weights - sem_weights,
                     mean_weights + sem_weights,
                     alpha=0.3, color='black')
    ax1.plot(neu_time, mean_weights_shuffled, label='Shuffled', color='gray', linestyle='--')
    ax1.fill_between(neu_time, 
                     mean_weights_shuffled - sem_weights_shuffled,
                     mean_weights_shuffled + sem_weights_shuffled,
                     alpha=0.3, color='gray')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Stim1 onset')
    ax1.set_ylabel('Mean |Weight| (across neurons)')
    ax1.set_title('SVM Feature Importance: Normal vs Shuffled')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ISI distributions
    if isi is not None and trial_type is not None:
        # Plot ISI distributions for short and long
        sns.kdeplot(x=isi[trial_type == 0] + 200, label='Short', color='blue', ax=ax2, bw_adjust=1.5)
        sns.kdeplot(x=isi[trial_type == 1] + 200, label='Long', color='red', ax=ax2, bw_adjust=1.5)
        ax2.legend()
    ax2.set_xlabel('Time')
    ax2.set_ylabel('ISI Density')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        figures_dir = os.path.join(save_path, 'Figures')
        os.makedirs(figures_dir, exist_ok=True)
        fig_path = os.path.join(figures_dir, f'average_weights.pdf')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
    plt.close()

def generate_decoder_labels(block_type, trial_type, outcomes=None, percentile=40, 
                          process_blocks='both', process_outcomes='both'):
    """
    Generate decoder labels for block trials based on early/late majority trials,
    with optional outcome filtering.
    
    Parameters:
    -----------
    block_type : array-like
        Array containing block types: 0 (undefined), 1 (short majority), 2 (long majority)
    trial_type : array-like
        Array containing trial types: 0 (short), 1 (long)
    outcomes : array-like, optional
        Array containing outcome strings (e.g., 'reward', 'punish', 'timeout', etc.)
        If None, no outcome filtering is applied
    percentile : float, default=25
        Percentage of majority trials to use for early and late labels
    process_blocks : str, default='both'
        Which blocks to process: 'short', 'long', or 'both'
    process_outcomes : str, default='both'
        Which outcome trials to process: 'reward', 'punish', or 'both'
        Only used if outcomes is provided
    
    Returns:
    --------
    labels : numpy.ndarray
        Array of labels: 0 (early trials), 1 (late trials), -1 (ignore)
    """
    
    block_type = np.array(block_type)
    trial_type = np.array(trial_type)
    labels = np.full(len(block_type), -1, dtype=int)  # Initialize all as ignore (-1)
    
    # Validate inputs
    if len(block_type) != len(trial_type):
        raise ValueError("block_type and trial_type must have the same length")
    
    if outcomes is not None:
        outcomes = np.array(outcomes)
        if len(outcomes) != len(block_type):
            raise ValueError("outcomes must have the same length as block_type and trial_type")
    
    if not 0 < percentile < 50:
        raise ValueError("percentile must be between 0 and 50")
    
    # Find unique blocks and their boundaries
    unique_blocks = np.unique(block_type)
    
    # Process each block type based on user preference
    blocks_to_process = []
    if process_blocks in ['short', 'both'] and 1 in unique_blocks:
        blocks_to_process.append(1)
    if process_blocks in ['long', 'both'] and 2 in unique_blocks:
        blocks_to_process.append(2)
    
    # Find block boundaries (where block_type changes)
    block_changes = np.where(np.diff(block_type) != 0)[0] + 1
    block_starts = np.concatenate([[0], block_changes])
    block_ends = np.concatenate([block_changes, [len(block_type)]])
    
    # Process each block
    for start, end in zip(block_starts, block_ends):
        current_block_type = block_type[start]
        
        # Skip if this block type is not in our processing list
        if current_block_type not in blocks_to_process:
            continue
        
        # Get trials in this block
        block_trials = trial_type[start:end]
        block_outcomes = outcomes[start:end] if outcomes is not None else None
        
        # Determine majority trial type for this block
        if current_block_type == 1:  # short majority block
            majority_trial_type = 0  # short trials
        elif current_block_type == 2:  # long majority block
            majority_trial_type = 1  # long trials
        else:
            continue  # Skip undefined blocks (type 0)
        
        # Find indices of majority trials within this block
        majority_mask = (block_trials == majority_trial_type)
        
        # Apply outcome filtering if provided
        if outcomes is not None:
            if process_outcomes == 'reward':
                outcome_mask = (block_outcomes == 'reward')
            elif process_outcomes == 'punish':
                outcome_mask = (block_outcomes == 'punish')
            elif process_outcomes == 'both':
                # Include both reward and punishment trials
                outcome_mask = (block_outcomes == 'reward') | (block_outcomes == 'punish')
            else:
                raise ValueError("process_outcomes must be 'reward', 'punish', or 'both'")
            
            # Combine majority and outcome masks
            combined_mask = majority_mask & outcome_mask
        else:
            combined_mask = majority_mask
        
        majority_indices = np.where(combined_mask)[0]
        
        if len(majority_indices) == 0:
            continue  # No valid trials in this block
        
        # Calculate number of trials for early and late labels
        num_majority_trials = len(majority_indices)
        num_early_late = max(1, int(num_majority_trials * percentile / 100))
        
        # Get early trials (first percentile of majority trials)
        early_indices = majority_indices[:num_early_late]
        early_global_indices = start + early_indices
        labels[early_global_indices] = 0
        
        # Get late trials (last percentile of majority trials)
        late_indices = majority_indices[-num_early_late:]
        late_global_indices = start + late_indices
        labels[late_global_indices] = 1
    
    return labels

def generate_majority_labels(block_type, trial_type, outcomes=None, process_blocks='both', process_outcomes='both'):
    """
    Generate labels for all majority trials in each block.
    
    Parameters:
    -----------
    block_type : array-like
        Array containing block types: 
        - 0 (neutral blocks - IGNORED)
        - 1 (short majority blocks)
        - 2 (long majority blocks)
    trial_type : array-like
        Array containing trial types: 0 (short), 1 (long)
    outcomes : array-like, optional
        Array containing outcome strings (e.g., 'reward', 'punish', 'timeout', etc.)
        If None, no outcome filtering is applied
    process_blocks : str, default='both'
        Which blocks to process: 'short', 'long', or 'both'
    process_outcomes : str, default='both'
        Which outcome trials to process: 'reward', 'punish', or 'both'
        Only used if outcomes is provided
    
    Returns:
    --------
    labels : numpy.ndarray
        Array of labels: 
        - 0 (short trials in short majority blocks)
        - 1 (long trials in long majority blocks)
        - -1 (ignore - neutral blocks, minority trials, or unprocessed blocks)
    """
    
    block_type = np.array(block_type)
    trial_type = np.array(trial_type)
    labels = np.full(len(block_type), -1, dtype=int)  # Initialize all as ignore (-1)
    
    # Validate inputs
    if len(block_type) != len(trial_type):
        raise ValueError("block_type and trial_type must have the same length")
    
    if outcomes is not None:
        outcomes = np.array(outcomes)
        if len(outcomes) != len(block_type):
            raise ValueError("outcomes must have the same length as block_type and trial_type")
    
    # Process each block type based on user preference
    blocks_to_process = []
    if process_blocks in ['short', 'both']:
        blocks_to_process.append(1)
    if process_blocks in ['long', 'both']:
        blocks_to_process.append(2)
    
    # Process each trial
    for i in range(len(block_type)):
        current_block_type = block_type[i]
        current_trial_type = trial_type[i]
        current_outcome = outcomes[i] if outcomes is not None else None
        
        # Skip if this block type is not in our processing list
        if current_block_type not in blocks_to_process:
            continue
        
        # Apply outcome filtering if provided
        if outcomes is not None:
            if process_outcomes == 'reward':
                if current_outcome != 'reward':
                    continue
            elif process_outcomes == 'punish':
                if current_outcome != 'punish':
                    continue
            elif process_outcomes == 'both':
                if current_outcome not in ['reward', 'punish']:
                    continue
            else:
                raise ValueError("process_outcomes must be 'reward', 'punish', or 'both'")
        
        # Label majority trials
        if current_block_type == 1 and current_trial_type == 0:
            # Short trial in short majority block
            labels[i] = 0
        elif current_block_type == 2 and current_trial_type == 1:
            # Long trial in long majority block  
            labels[i] = 1
        # All other cases remain -1 (ignore)
    
    return labels

def generate_majority_rare_labels(block_type, trial_type, outcomes=None, process_blocks='both', process_outcomes='both'):
    """
    Generate labels for majority vs rare trials based on block-trial type combinations.
    
    Parameters:
    -----------
    block_type : array-like
        Array containing block types: 
        - 0 (neutral blocks - IGNORED)
        - 1 (short majority blocks)
        - 2 (long majority blocks)
    trial_type : array-like
        Array containing trial types: 0 (short), 1 (long)
    outcomes : array-like, optional
        Array containing outcome strings (e.g., 'reward', 'punish', 'timeout', etc.)
        If None, no outcome filtering is applied
    process_blocks : str, default='both'
        Which trial types to process: 'short', 'long', or 'both'
        - 'short': Only process short trials (both common in short blocks and rare in long blocks)
        - 'long': Only process long trials (both common in long blocks and rare in short blocks)
        - 'both': Process both trial types
    process_outcomes : str, default='both'
        Which outcome trials to process: 'reward', 'punish', or 'both'
        Only used if outcomes is provided
    
    Returns:
    --------
    labels : numpy.ndarray
        Array of labels: 
        - 0 (majority trials: short trials in short blocks, long trials in long blocks)
        - 1 (rare trials: short trials in long blocks, long trials in short blocks)
        - -1 (ignore - neutral blocks or unprocessed blocks/outcomes)
    """
    
    block_type = np.array(block_type)
    trial_type = np.array(trial_type)
    labels = np.full(len(block_type), -1, dtype=int)  # Initialize all as ignore (-1)
    
    # Validate inputs
    if len(block_type) != len(trial_type):
        raise ValueError("block_type and trial_type must have the same length")
    
    if outcomes is not None:
        outcomes = np.array(outcomes)
        if len(outcomes) != len(block_type):
            raise ValueError("outcomes must have the same length as block_type and trial_type")
    
    # Determine which trial types to process based on user preference
    trial_types_to_process = []
    if process_blocks in ['short', 'both']:
        trial_types_to_process.append(0)  # Process short trials
    if process_blocks in ['long', 'both']:
        trial_types_to_process.append(1)  # Process long trials
    
    # Process each trial
    for i in range(len(block_type)):
        current_block_type = block_type[i]
        current_trial_type = trial_type[i]
        current_outcome = outcomes[i] if outcomes is not None else None
        
        # Skip neutral blocks
        if current_block_type == 0:
            continue
        
        # Skip if this trial type is not in our processing list
        if current_trial_type not in trial_types_to_process:
            continue
        
        # Apply outcome filtering if provided
        if outcomes is not None:
            if process_outcomes == 'reward':
                if current_outcome != 'reward':
                    continue
            elif process_outcomes == 'punish':
                if current_outcome != 'punish':
                    continue
            elif process_outcomes == 'both':
                if current_outcome not in ['reward', 'punish']:
                    continue
            else:
                raise ValueError("process_outcomes must be 'reward', 'punish', or 'both'")
        
        # Label majority vs rare trials
        if current_block_type == 1 and current_trial_type == 0:
            # Short trial in short majority block (majority/common)
            labels[i] = 0
        elif current_block_type == 1 and current_trial_type == 1:
            # Long trial in short majority block (rare)
            labels[i] = 1
        elif current_block_type == 2 and current_trial_type == 0:
            # Short trial in long majority block (rare)
            labels[i] = 1
        elif current_block_type == 2 and current_trial_type == 1:
            # Long trial in long majority block (majority/common)
            labels[i] = 0
        # All other cases remain -1 (ignore)
    
    return labels

def separate_rare_trials(block_type, trial_type, outcome, separation_type='both'):
    """
    Separates rare trials from each block type and labels their outcomes.
    
    Parameters:
    - block_type: List with 0, 1, 2 (where 1 = short majority, 2 = long majority)
    - trial_type: List with 0, 1 (where 0 = short trial, 1 = long trial)
    - outcome: List with trial outcomes (e.g., 'reward', 'punish')
    - separation_type: String ('both', 'short', 'long') to specify which trials to separate
    
    Returns:
    - List of labeled outcomes for all trials (-1 for not desired trials, 0 for reward rare trials, 1 for punish rare trials)
    """
    # Validate inputs
    if len(block_type) != len(trial_type) or len(block_type) != len(outcome):
        raise ValueError("All arrays must have the same length")
    
    results = []
    
    for i in range(len(block_type)):
        current_block_type = block_type[i]
        current_trial_type = trial_type[i]
        current_outcome = outcome[i]
        
        # Default to -1 for not desired trials
        labeled_outcome = -1
        
        # Only process if block type is not 0 (undefined/neutral block)
        if current_block_type != 0:
            is_rare_trial = False
            
            # Determine if current trial is rare based on separation type
            if separation_type == 'both':
                # For short majority blocks (1): long trials (1) are rare
                # For long majority blocks (2): short trials (0) are rare
                if current_block_type == 1 and current_trial_type == 1:
                    is_rare_trial = True  # Long trial in short majority block
                elif current_block_type == 2 and current_trial_type == 0:
                    is_rare_trial = True  # Short trial in long majority block
            elif separation_type == 'short':
                # Only separate long trials (1) as rare, regardless of block type
                if current_trial_type == 1 and current_block_type==1:
                    is_rare_trial = True
            elif separation_type == 'long':
                # Only separate short trials (0) as rare, regardless of block type
                if current_trial_type == 0 and current_block_type == 2:
                    is_rare_trial = True
            
            # If this is a rare trial, label the outcome
            if is_rare_trial:
                if current_outcome == 'reward':
                    labeled_outcome = 0
                elif current_outcome == 'punish':
                    labeled_outcome = 1
                # If outcome is neither 'reward' nor 'punish', keep as -1
        
        results.append(labeled_outcome)
    
    return np.array(results)

def decoder_decision(neural_trials, labels, decoding, indice, l_frames=60, r_frames=120, 
                                           save_path=None, pooling=False):
    """
    Modified decoder_decision function that includes temporal decoding analysis. decoding eraly trials of blocks form the last trials of blocks.
    This function pools data from multiple sessions, runs the original SVM decoder,
    """
    
    if pooling:
        neu_seq, neu_time, trial_type, block_type, isi, decision, labels, outcome = pool_session_data(
            neural_trials, labels, 'stim_seq', l_frames, r_frames, indices=indice)
    else:
        neu_seq, neu_time, stim_seq, stim_value, stim_time, led_value, trial_type, block_type, isi, decision, outcome = get_perception_response(
            neural_trials, 'stim_seq', l_frames, r_frames, indices=indice)

    if decoding == 'epoch_short':
        # trial_type: 0 for early, 1 for late
        labels_decoder = generate_decoder_labels(block_type, trial_type, outcomes=outcome, percentile=40, process_blocks='short', process_outcomes='both')
        label_names = ['Early', 'Late']
    elif decoding == 'epoch_long':
        labels_decoder = generate_decoder_labels(block_type, trial_type, outcomes=outcome, percentile=40, process_blocks='long', process_outcomes='both')
        label_names = ['Early', 'Late']
    elif decoding == 'trial_type':
        # trial_type: 0 for short, 1 for long
        labels_decoder = trial_type
        label_names = ['Short', 'long']
    elif decoding == 'block_type':
        labels_decoder = generate_majority_labels(block_type, trial_type, outcomes=outcome, process_blocks='both', process_outcomes='punish')
        label_names = ['Short', 'long']
    elif decoding == 'rare vs common':
        labels_decoder = generate_majority_rare_labels(block_type, trial_type, outcomes=outcome, process_blocks='short', process_outcomes='both')
        label_names = ['Short_common', 'Short_rare']
        # label_names = ['Long_common', 'Long_rare']
    elif decoding == 'rare':
        labels_decoder = separate_rare_trials(block_type, trial_type, outcome, separation_type='long')
        label_names = ['correct', 'incorrect']
    elif decoding == 'side':
        # left or right lick
        labels_decoder = decision
        label_names = ['Left', 'Right']

    # Run original decoder analysis
    results_unbinned = neural_decoder_svm(neu_seq, labels_decoder, neu_time, trial_type, isi, label_names,
                                        bin_data=False, save_path=save_path)

    # # Shuffle labels analysis
    labels_decoder_shuffle = np.random.permutation(labels_decoder)
    results_unbinned_shuffled = neural_decoder_svm(neu_seq, labels_decoder_shuffle, neu_time, 
                                                 trial_type, isi,label_names, shuffle='shuffle', 
                                                 bin_data=False, save_path=save_path)

    plot_shuffle_vs_normal(results_unbinned, results_unbinned_shuffled, neu_time, isi, 
                          trial_type, save_path=save_path)

    # NEW: Temporal decoding analysis
    print("\n=== Running Temporal Decoding Analysis ===")
    temporal_results = temporal_decoding_accuracy(neu_seq, labels_decoder, neu_time, 
                                                trial_type, isi, save_path=save_path)
    
    plot_temporal_decoding_accuracy(temporal_results, isi, trial_type, decoding, save_path=save_path)

    # Access results
    print(f"Best CV accuracy: {results_unbinned['cv_mean']:.3f}")
    print(f"Peak temporal decoding accuracy: {np.max(temporal_results['mean_acc_normal']):.3f}")
    
    return temporal_results