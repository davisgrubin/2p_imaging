#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

from modeling.utils import norm01
from modeling.utils import get_mean_sem

# clustering neural response.
def clustering_neu_response_mode(x_in, n_clusters, max_clusters):
    # cluster number must be less than sample number.
    n_clusters = n_clusters if n_clusters < x_in.shape[0] else x_in.shape[0]
    max_clusters = max_clusters if max_clusters < x_in.shape[0] else x_in.shape[0]
    # compute evaluation metrics.
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    inertia_values = []
    for n in tqdm(range(2, max_clusters+1)):
        model = KMeans(n_clusters=n)
        cluster_id = model.fit_predict(x_in)
        silhouette_scores.append(silhouette_score(x_in, cluster_id))
        calinski_harabasz_scores.append(calinski_harabasz_score(x_in, cluster_id))
        davies_bouldin_scores.append(davies_bouldin_score(x_in, cluster_id))
        inertia_values.append(model.inertia_)
    metrics = {
        'n_clusters': np.arange(2,max_clusters+1),
        'silhouette': norm01(np.array(silhouette_scores)),
        'calinski_harabasz': norm01(np.array(calinski_harabasz_scores)),
        'davies_bouldin': norm01(np.array(davies_bouldin_scores)),
        'inertia': norm01(np.array(inertia_values)),
        }
    # run clustering model.
    model = KMeans(n_clusters)
    cluster_id = model.fit_predict(x_in)
    # relabel based on the number of elements.
    unique, counts = np.unique(cluster_id, return_counts=True)
    sorted_labels = unique[np.argsort(-counts)]
    mapping = {val: i for i, val in enumerate(sorted_labels)}
    cluster_id = np.vectorize(mapping.get)(cluster_id)
    return metrics, cluster_id

# compute mean and sem for clusters.
def get_mean_sem_cluster(neu, cluster_id):
    neu_mean = np.zeros((len(np.unique(cluster_id)), neu.shape[1]))
    neu_sem  = np.zeros((len(np.unique(cluster_id)), neu.shape[1]))
    for i in range(len(np.unique(cluster_id))):
        neu_mean[i,:], neu_sem[i,:] = get_mean_sem(
            neu[np.where(cluster_id==i)[0], :].reshape(-1,neu.shape[1]))
    return neu_mean, neu_sem

# compute sorted correlation matrix.
def get_sorted_corr_mat(neu, cluster_id):
    neu_corr = np.corrcoef(neu)
    sorted_indices = np.argsort(cluster_id)
    sorted_neu_corr = neu_corr[sorted_indices, :][:, sorted_indices]
    return sorted_neu_corr

# compute cross cluster correlations.
def get_cross_corr(neu, n_clusters, cluster_id):
    neu_mean, _ = get_mean_sem_cluster(neu, cluster_id)
    cluster_corr = np.corrcoef(neu_mean)
    return cluster_corr


