from Modules.Alignment import get_perception_response
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import pandas as pd
from scipy import stats
import os
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

class NeuralActivityClustering:
    def __init__(self, neural_data, random_state=42):
        """
        Initialize clustering analysis for neural activity data
        
        Parameters:
        neural_data: numpy array of shape (n_neurons, time_points)
                    Average neural activity from first stim onset to second stim onset
        """
        self.neural_data = neural_data
        self.n_neurons, self.n_timepoints = neural_data.shape
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.best_gmm = None
        self.cluster_labels = None
        self.cv_results = {}
        
    def preprocess_data(self, apply_pca=True, n_components=0.95):
        """
        Preprocess neural activity data with z-score normalization and optional PCA
        
        Parameters:
        apply_pca: bool, whether to apply PCA for dimensionality reduction
        n_components: float or int, number of components to keep (if float, variance explained)
        """
        print("Preprocessing neural activity data...")
        
        # Z-score normalization across time for each neuron
        self.data_normalized = self.scaler.fit_transform(self.neural_data.T).T
        
        if apply_pca:
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            self.data_processed = self.pca.fit_transform(self.data_normalized)
            print(f"PCA applied: {self.data_processed.shape[1]} components explain "
                  f"{self.pca.explained_variance_ratio_.sum():.3f} of variance")
        else:
            self.data_processed = self.data_normalized
            
        return self.data_processed
    
    def find_optimal_clusters(self, max_clusters=15, cv_folds=5):
        """
        Find optimal number of clusters using cross-validation and multiple metrics
        
        Parameters:
        max_clusters: int, maximum number of clusters to test
        cv_folds: int, number of cross-validation folds
        """
        print(f"Finding optimal number of clusters (max={max_clusters})...")
        
        n_clusters_range = range(2, max_clusters + 1)
        
        # Initialize result storage
        results = {
            'n_clusters': [],
            'bic_mean': [], 'bic_std': [],
            'aic_mean': [], 'aic_std': [],
            'silhouette_mean': [], 'silhouette_std': [],
            'calinski_harabasz_mean': [], 'calinski_harabasz_std': [],
            'cv_stability_mean': [], 'cv_stability_std': []
        }
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for n_clusters in n_clusters_range:
            # print(f"Testing {n_clusters} clusters...")
            
            fold_bic, fold_aic, fold_silhouette, fold_calinski = [], [], [], []
            fold_labels = []
            
            # Cross-validation
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.data_processed)):
                # Fit GMM on training data
                gmm = GaussianMixture(
                    n_components=n_clusters,
                    covariance_type='diag',  # Changed from 'full' to 'diag'
                    reg_covar=1e-6,          # Added regularization
                    random_state=self.random_state + fold,
                    max_iter=200,
                    n_init=5                 # Multiple initializations
                )
                gmm.fit(self.data_processed[train_idx])
                
                # Predict on validation data
                val_labels = gmm.predict(self.data_processed[val_idx])
                fold_labels.append(val_labels)
                
                # Calculate metrics on validation set
                fold_bic.append(gmm.bic(self.data_processed[val_idx]))
                fold_aic.append(gmm.aic(self.data_processed[val_idx]))
                
                if len(np.unique(val_labels)) > 1:  # Check if more than one cluster
                    fold_silhouette.append(silhouette_score(self.data_processed[val_idx], val_labels))
                    fold_calinski.append(calinski_harabasz_score(self.data_processed[val_idx], val_labels))
                else:
                    fold_silhouette.append(-1)  # Invalid score
                    fold_calinski.append(0)
            
            # Calculate stability across folds (using ARI between fold pairs)
            stability_scores = []
            for i in range(len(fold_labels)):
                for j in range(i+1, len(fold_labels)):
                    if len(fold_labels[i]) == len(fold_labels[j]):
                        stability_scores.append(adjusted_rand_score(fold_labels[i], fold_labels[j]))
            
            # Store results
            results['n_clusters'].append(n_clusters)
            results['bic_mean'].append(np.mean(fold_bic))
            results['bic_std'].append(np.std(fold_bic))
            results['aic_mean'].append(np.mean(fold_aic))
            results['aic_std'].append(np.std(fold_aic))
            results['silhouette_mean'].append(np.mean(fold_silhouette))
            results['silhouette_std'].append(np.std(fold_silhouette))
            results['calinski_harabasz_mean'].append(np.mean(fold_calinski))
            results['calinski_harabasz_std'].append(np.std(fold_calinski))
            results['cv_stability_mean'].append(np.mean(stability_scores) if stability_scores else 0)
            results['cv_stability_std'].append(np.std(stability_scores) if stability_scores else 0)
        
        self.cv_results = pd.DataFrame(results)
        
        # Find optimal number of clusters
        # Combine multiple criteria (lower BIC/AIC, higher silhouette/calinski/stability)
        normalized_bic = (np.max(results['bic_mean']) - np.array(results['bic_mean'])) / \
                        (np.max(results['bic_mean']) - np.min(results['bic_mean']))
        normalized_aic = (np.max(results['aic_mean']) - np.array(results['aic_mean'])) / \
                        (np.max(results['aic_mean']) - np.min(results['aic_mean']))
        normalized_silhouette = (np.array(results['silhouette_mean']) - np.min(results['silhouette_mean'])) / \
                               (np.max(results['silhouette_mean']) - np.min(results['silhouette_mean']))
        normalized_stability = (np.array(results['cv_stability_mean']) - np.min(results['cv_stability_mean'])) / \
                              (np.max(results['cv_stability_mean']) - np.min(results['cv_stability_mean']))
        
        # Composite score (equal weights)
        composite_score = (normalized_bic + normalized_aic + normalized_silhouette + normalized_stability) / 4
        self.optimal_n_clusters = n_clusters_range[np.argmax(composite_score)]
        
        print(f"Optimal number of clusters: {self.optimal_n_clusters}")
        return self.optimal_n_clusters
    
    def fit_final_model(self, n_clusters=None):
        """
        Fit final GMM model with optimal number of clusters
        
        Parameters:
        n_clusters: int, number of clusters (if None, uses optimal from CV)
        """
        if n_clusters is None:
            n_clusters = self.optimal_n_clusters
            
        print(f"Fitting final GMM model with {n_clusters} clusters...")
        
        self.best_gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='diag',  # Changed from 'full' to 'diag'
            reg_covar=1e-6,          # Added regularization
            random_state=self.random_state,
            max_iter=200,
            n_init=10                # Multiple initializations
        )
        
        self.best_gmm.fit(self.data_processed)
        self.cluster_labels = self.best_gmm.predict(self.data_processed)
        self.cluster_probabilities = self.best_gmm.predict_proba(self.data_processed)
        
        # Calculate final metrics
        self.final_metrics = {
            'bic': self.best_gmm.bic(self.data_processed),
            'aic': self.best_gmm.aic(self.data_processed),
            'silhouette': silhouette_score(self.data_processed, self.cluster_labels),
            'calinski_harabasz': calinski_harabasz_score(self.data_processed, self.cluster_labels),
            'log_likelihood': self.best_gmm.score(self.data_processed)
        }
        
        print("Final clustering metrics:")
        for metric, value in self.final_metrics.items():
            print(f"  {metric}: {value:.3f}")
            
        return self.cluster_labels
    
    def plot_cv_results(self, figsize=(15, 10), save_path=None):
        """Plot cross-validation results"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.ravel()
        
        metrics = [
            ('BIC', 'bic_mean', 'bic_std', 'lower is better'),
            ('AIC', 'aic_mean', 'aic_std', 'lower is better'),
            ('Silhouette Score', 'silhouette_mean', 'silhouette_std', 'higher is better'),
            ('Calinski-Harabasz', 'calinski_harabasz_mean', 'calinski_harabasz_std', 'higher is better'),
            ('CV Stability (ARI)', 'cv_stability_mean', 'cv_stability_std', 'higher is better')
        ]
        
        for i, (title, mean_col, std_col, direction) in enumerate(metrics):
            axes[i].errorbar(self.cv_results['n_clusters'], 
                           self.cv_results[mean_col], 
                           yerr=self.cv_results[std_col],
                           marker='o', capsize=5)
            axes[i].axvline(x=self.optimal_n_clusters, color='red', linestyle='--', alpha=0.7)
            axes[i].set_xlabel('Number of Clusters')
            axes[i].set_ylabel(title)
            axes[i].set_title(f'{title} ({direction})')
            # axes[i].grid(True, alpha=0.3)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
        
        # Remove the last subplot
        axes[-1].remove()
        
        plt.tight_layout()
        if save_path is not None:
            figures_dir = os.path.join(save_path, 'Figures')
            if not os.path.exists(figures_dir):
                os.makedirs(figures_dir)
            save_path = os.path.join(figures_dir, 'clustering_metrics.pdf')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_clustering_results(self, neu_time, n_neurons, figsize=(20, 12), save_path=None):

        def apply_colormap(data, cmap):
            """Normalize data and apply colormap."""
            if len(data) > 0:
                vmin, vmax = data.min(), data.max()
                normalized = (data - vmin) / (vmax - vmin + 1e-10)
                return cmap(normalized)
            return np.array([])

        """Plot clustering results and neural activity patterns"""
        n_clusters = len(np.unique(self.cluster_labels))

        fig = plt.figure(figsize=figsize)

        # 1. PCA visualization
        if self.pca is not None:
            ax1 = plt.subplot(2, 3, 1)
            scatter = ax1.scatter(self.data_processed[:, 0], self.data_processed[:, 1],
                                c=self.cluster_labels, cmap='tab10', alpha=0.7)
            ax1.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.3f} variance)')
            ax1.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.3f} variance)')
            ax1.set_title('Clusters in PCA Space')
            fig.colorbar(scatter, ax=ax1)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

        # 2. Cluster probability heatmap
        ax2 = plt.subplot(2, 3, 2)
        sorted_idx = np.argsort(self.cluster_labels)
        ax2.imshow(self.cluster_probabilities[sorted_idx].T, aspect='auto', cmap='viridis')
        ax2.set_xlabel('Neuron (sorted by cluster)')
        ax2.set_ylabel('Cluster')
        ax2.set_title('Cluster Membership Probabilities')
        plt.colorbar(ax2.images[0], ax=ax2)

        # 3. Average neural activity per cluster
        ax3 = plt.subplot(2, 3, 3)
        for cluster_id in range(n_clusters):
            cluster_neurons = self.cluster_labels == cluster_id
            mean_activity = np.mean(self.neural_data[cluster_neurons], axis=0)
            sem_activity = stats.sem(self.neural_data[cluster_neurons], axis=0)

            ax3.plot(neu_time, mean_activity, label=f'Cluster {cluster_id}', linewidth=2)
            ax3.fill_between(neu_time,
                            mean_activity - sem_activity,
                            mean_activity + sem_activity,
                            alpha=0.3)
        ax3.set_xlabel('Time (from first stim onset)')
        ax3.set_ylabel('Neural Activity')
        ax3.set_title('Average Activity per Cluster')
        ax3.legend()
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # 4. Cluster sizes
        ax4 = plt.subplot(2, 3, 4)
        cluster_counts = np.bincount(self.cluster_labels)
        ax4.bar(range(n_clusters), cluster_counts, color=plt.cm.tab10(np.arange(n_clusters)))
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Number of Neurons')
        ax4.set_title('Cluster Sizes')

        # 5. Neural activity heatmap sorted by cluster
        ax5 = plt.subplot(2, 3, 5)
        sorted_data = self.neural_data[sorted_idx]
        cmap = cm.get_cmap('inferno')
        heatmap_rgb = np.array([apply_colormap(row, cmap) for row in sorted_data])
        im = ax5.imshow(heatmap_rgb, interpolation='nearest', aspect='auto',
                   extent=[neu_time[0], neu_time[-1], n_neurons, 0])
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Neuron (sorted by cluster)')
        ax5.set_title('Neural Activity Heatmap')
        # plt.colorbar(im, ax=ax5)

        # Add cluster boundaries
        cluster_boundaries = np.cumsum(cluster_counts)[:-1]
        for boundary in cluster_boundaries:
            ax5.axhline(y=boundary - 0.5, color='white', linewidth=2)

        # Remove last (empty) subplot if figure has more slots than used
        # if len(fig.axes) > 5:
        #     fig.delaxes(fig.axes[-1])

        plt.tight_layout()

        if save_path is not None:
            figures_dir = os.path.join(save_path, 'Figures')
            os.makedirs(figures_dir, exist_ok=True)
            fig_path = os.path.join(figures_dir, 'clustering_results.pdf')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()


def main(neural_data, neu_time, save_path=None):
    
    # Initialize and run clustering
    clustering = NeuralActivityClustering(neural_data)
    
    # Preprocess data
    clustering.preprocess_data(apply_pca=True, n_components=0.95)
    
    # Find optimal number of clusters
    optimal_k = clustering.find_optimal_clusters(max_clusters=15, cv_folds=10)
    
    # Fit final model
    cluster_labels = clustering.fit_final_model()
    
    # Plot results
    clustering.plot_cv_results(save_path=save_path)
    n_neurons = neural_data.shape[0]
    clustering.plot_clustering_results(neu_time, n_neurons, save_path=save_path)

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

def find_min_long_trial_isi(neural_trials):
    """
    Find the smallest ISI (inter-stimulus interval) among long trials (trial_type == 1).

    Parameters:
    neural_trials (dict or list of dict): Single session or list of sessions. 
        Each session should be a dict with key ['trial_labels'] containing keys 'trial_type' and 'isi'.

    Returns:
    float: Minimum ISI among long trials across all sessions.
    """
    min_time_interval = np.inf

    if isinstance(neural_trials, dict):
        # Single session
        long_trials = neural_trials['trial_labels']['trial_type'] == 1
        if np.any(long_trials):
            min_time_interval = np.min(neural_trials['trial_labels']['isi'][long_trials])
    elif isinstance(neural_trials, list):
        # Multiple sessions
        for session in neural_trials:
            long_trials = session['trial_labels']['trial_type'] == 1
            if np.any(long_trials):
                session_min = np.min(session['trial_labels']['isi'][long_trials])
                min_time_interval = min(min_time_interval, session_min)

    if np.isinf(min_time_interval):
        raise ValueError("No long trials (trial_type == 1) found in the provided data.")
    
    # turn it to the timepoint
    # fs = 30 hz
    min_time_interval_frame = int(np.floor(min_time_interval / 30))

    return min_time_interval, min_time_interval_frame

def get_avg_neural_activity(neu_seq, trial_type, block_type):
    """
    Compute average neural activity from first stim onset to second stim onset for each neuron,
    using only long trials of short block (trial_type == 1, bloc_type == 1).

    Parameters:
    neu_seq: np.ndarray, shape (n_trials, n_neurons, time)
    trial_type: np.ndarray, shape (n_trials,)

    Returns:
    avg_neuron: np.ndarray, shape (n_neurons, time)
    """
    # Select long trials (trial_type == 1) of short block (block_type == 1)
    # Note: block_type must be passed as an argument
    long_trials = (trial_type == 1) & (block_type == 1)
    avg_neuron = np.nanmean(neu_seq[long_trials], axis=0)  # Shape: (n_neurons, time)
    return avg_neuron
    
def clustering_GMM(neural_trials, labels, save_path = None, pooling = False):
    
    # get the framse for the smallest time interval among the long trials
    l_frames = 0
    _ , r_frames = find_min_long_trial_isi(neural_trials)
    if pooling:
        neu_seq, neu_time, trial_type, block_type, isi, decision, labels, outcome = pool_session_data(neural_trials, labels, 'stim_seq', l_frames, r_frames, indices=0)
    else:
        neu_seq, neu_time, stim_seq, stim_value, stim_time, led_value, trial_type, block_type, isi, decision, outcome = get_perception_response(neural_trials, 'stim_seq', l_frames, r_frames, indices=0)

    # Average neural activity from first stim onset to second stim onset for each neuron
    neural_data = get_avg_neural_activity(neu_seq, trial_type, block_type)
    main(neural_data,neu_time, save_path = save_path)
