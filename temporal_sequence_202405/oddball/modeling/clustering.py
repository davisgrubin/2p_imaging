#!/usr/bin/env python3

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

from plot.utils import norm01
from plot.utils import get_mean_sem

# clustering neural response.
def clustering_neu_response_mode(
        neu, n_clusters, max_clusters,
        l_idx, r_idx
        ):
    # compute evaluation metrics.
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    inertia_values = []
    for n in range(2, max_clusters):
        corr_matrix = np.corrcoef(neu)
        model = KMeans(n_clusters=n)
        cluster_id = model.fit_predict(corr_matrix)
        silhouette_scores.append(silhouette_score(neu, cluster_id))
        calinski_harabasz_scores.append(calinski_harabasz_score(neu, cluster_id))
        davies_bouldin_scores.append(davies_bouldin_score(neu, cluster_id))
        inertia_values.append(model.inertia_)
    metrics = {
        'n_clusters': np.arange(2,max_clusters),
        'silhouette': norm01(np.array(silhouette_scores)),
        'calinski_harabasz': norm01(np.array(calinski_harabasz_scores)),
        'davies_bouldin': norm01(np.array(davies_bouldin_scores)),
        'inertia': norm01(np.array(inertia_values)),
        }
    neu_corr = np.corrcoef(neu[:,l_idx:r_idx])
    model = KMeans(n_clusters)
    cluster_id = model.fit_predict(neu_corr)
    return metrics, cluster_id

# compute sorted correlation matrix.
def get_sorted_corr_mat(neu, cluster_id, l_idx, r_idx):
    neu_corr = np.corrcoef(neu[:,l_idx:r_idx])
    sorted_indices = np.argsort(cluster_id)
    sorted_neu_corr = neu_corr[sorted_indices, :][:, sorted_indices]
    return sorted_neu_corr

# compute mean and sem for clusters.
def get_mean_sem_cluster(neu, n_clusters, cluster_id, l_idx, r_idx):
    neu_mean = np.zeros((n_clusters, neu.shape[1]))
    neu_sem  = np.zeros((n_clusters, neu.shape[1]))
    for i in range(n_clusters):
        neu_mean[i,:], neu_sem[i,:] = get_mean_sem(
            neu[np.where(cluster_id==i)[0], :].reshape(-1,neu.shape[1]))
    return neu_mean, neu_sem

# compute cross cluster correlations.
def get_cross_corr(neu, n_clusters, cluster_id, l_idx, r_idx):
    neu_mean, _ = get_mean_sem_cluster(
        neu, n_clusters, cluster_id, l_idx, r_idx)
    cluster_corr = np.corrcoef(neu_mean)
    return cluster_corr


