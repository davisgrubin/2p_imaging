"""GLM helper functions extracted from `test_GLM.ipynb`."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit


def construct_design_matrix_for_stimulus(stimulus, kernel_length, pre_points):
    """Construct a lagged design matrix for a single stimulus trace."""
    n_timepoints = len(stimulus)
    if kernel_length <= 0 or pre_points < 0 or pre_points >= kernel_length:
        raise ValueError('Invalid kernel_length or pre_points.')

    post_points = kernel_length - pre_points
    lags = range(-pre_points, post_points)
    design_matrix = np.hstack([np.roll(stimulus, lag).reshape(-1, 1) for lag in lags])

    for i, lag in enumerate(lags):
        if lag > 0:
            design_matrix[:lag, i] = 0
        elif lag < 0:
            design_matrix[lag:, i] = 0

    return design_matrix


def create_design_matrix(stimuli, kernel_lengths, pre_points_list, include_offset=True):
    """Create a full design matrix with potentially different kernels per regressor."""
    if not (len(stimuli) == len(kernel_lengths) == len(pre_points_list)):
        raise ValueError('stimuli, kernel_lengths, and pre_points_list must have the same length.')

    design_matrices = [
        construct_design_matrix_for_stimulus(stim, k_len, pre_pts).astype(np.float32)
        for stim, k_len, pre_pts in zip(stimuli, kernel_lengths, pre_points_list)
    ]
    full_design_matrix = np.hstack(design_matrices)

    if include_offset:
        offset_column = np.ones((full_design_matrix.shape[0], 1), dtype=np.float32)
        full_design_matrix = np.hstack([offset_column, full_design_matrix])

    return full_design_matrix


def ridge_regression_bayesian_optimization(W, y_neuron, n_splits=5, n_calls=50):
    """Tune ridge regularization by Bayesian optimization over time-series CV."""
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args

    space = [Real(1e-2, 6e2, 'log-uniform', name='reg_lambda')]
    lambda_scores = []

    @use_named_args(space)
    def objective(reg_lambda):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_scores = []

        for train_idx, test_idx in tscv.split(W):
            W_train, W_test = W[train_idx], W[test_idx]
            y_train, y_test = y_neuron[train_idx], y_neuron[test_idx]

            I = np.eye(W_train.shape[1])
            try:
                weights = np.linalg.solve(W_train.T @ W_train + reg_lambda * I, W_train.T @ y_train)
                y_pred = W_test @ weights
                fold_scores.append(r2_score(y_test, y_pred))
            except np.linalg.LinAlgError:
                fold_scores.append(-np.inf)

        mean_score = np.mean(fold_scores)
        lambda_scores.append((reg_lambda, mean_score))
        return -mean_score

    result = gp_minimize(func=objective, dimensions=space, n_calls=n_calls, random_state=42)
    best_lambda = result.x[0]
    I = np.eye(W.shape[1])
    weights = np.linalg.solve(W.T @ W + best_lambda * I, W.T @ y_neuron)
    return best_lambda, weights, lambda_scores


def compute_explained_variance(y_neuron, W, weights, kernel_lengths, include_offset=True):
    """Compute overall and per-feature explained variance via feature dropout."""
    y_pred_full = W @ weights
    explained_variance_full = r2_score(y_neuron, y_pred_full)

    explained_variance_features = []
    current_col_idx = 1 if include_offset else 0

    for k_len in kernel_lengths:
        feature_indices = np.arange(current_col_idx, current_col_idx + k_len)
        W_reduced = np.delete(W, feature_indices, axis=1)
        weights_reduced = np.delete(weights, feature_indices)
        y_pred_reduced = W_reduced @ weights_reduced
        explained_variance_reduced = r2_score(y_neuron, y_pred_reduced)
        explained_variance_features.append(explained_variance_full - explained_variance_reduced)
        current_col_idx += k_len

    return explained_variance_full, explained_variance_features


def fit_glm_for_neuron(y_neuron, stimuli, kernel_lengths, pre_points_list):
    """Fit the full GLM pipeline for one neuron trace."""
    W = create_design_matrix(stimuli, kernel_lengths, pre_points_list)
    best_lambda, weights, lambda_scores = ridge_regression_bayesian_optimization(W, y_neuron)
    explained_variance_full, explained_variance_features = compute_explained_variance(
        y_neuron, W, weights, kernel_lengths
    )
    return {
        'weights': weights,
        'explained_variance_full': explained_variance_full,
        'explained_variance_features': explained_variance_features,
        'best_lambda': best_lambda,
        'lambda_scores': lambda_scores,
    }


def cluster_kernels(kernel_name, all_results, kernel_config, n_clusters_range=range(2, 8)):
    """PCA + GMM clustering for one set of fitted GLM kernels."""
    if kernel_name not in kernel_config:
        raise ValueError(f"Kernel '{kernel_name}' not found in kernel_config.")

    start_idx = 1
    target_k_len = 0
    for name, config in kernel_config.items():
        k_len = config['kernel_length']
        if name == kernel_name:
            target_k_len = k_len
            break
        start_idx += k_len

    kernels_matrix = np.array([res['weights'][start_idx:start_idx + target_k_len] for res in all_results])
    pca = PCA(n_components=0.95, random_state=42)
    pca_data = pca.fit_transform(kernels_matrix)

    bic_scores = []
    for k in n_clusters_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        gmm.fit(pca_data)
        bic_scores.append(gmm.bic(pca_data))

    best_k = n_clusters_range[np.argmin(bic_scores)]
    final_gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=10)
    cluster_labels = final_gmm.fit_predict(pca_data)

    return {
        'labels': cluster_labels,
        'best_k': best_k,
        'raw_kernels': kernels_matrix,
        'pca_data': pca_data,
        'pca_model': pca,
        'bic_scores': bic_scores,
        'n_clusters_range': n_clusters_range,
        'kernel_name': kernel_name,
    }
