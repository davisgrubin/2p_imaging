#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

from modeling.quantifications import run_quantification
from modeling.utils import norm01
from modeling.utils import get_mean_sem

# clustering neural response.
def clustering_neu_response_mode(x_in, n_clusters, max_clusters):
    # feature normalization.
    x_in -= np.nanmean(x_in, axis=1, keepdims=True)
    # cluster number must be less than sample number.
    n_clusters = n_clusters if n_clusters < x_in.shape[0] else x_in.shape[0]
    # compute evaluation metrics if needed.
    if max_clusters != None:
        max_clusters = max_clusters if max_clusters < x_in.shape[0] else x_in.shape[0]
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
    else:
        metrics = None
    # run clustering model.
    model = KMeans(n_clusters)
    cluster_id = model.fit_predict(x_in)
    return metrics, cluster_id

# organize cluster labels based on stimulus evoked latency.
def remap_cluster_id(neu, neu_time, cluster_id):
    neu_seq, _ = get_mean_sem_cluster(neu, cluster_id)
    evoke_time = run_quantification(neu_seq, neu_time, win_eval_c=0, samping_size=0)['evoke_latency']
    sorted_labels = np.argsort(evoke_time)[::-1]
    mapping = {val: i for i, val in enumerate(sorted_labels)}
    cluster_id = np.vectorize(mapping.get)(cluster_id)
    return cluster_id

# compute mean and sem for clusters.
def get_mean_sem_cluster(neu, cluster_id):
    neu_mean = np.zeros((len(np.unique(cluster_id)), neu.shape[1]))
    neu_sem  = np.zeros((len(np.unique(cluster_id)), neu.shape[1]))
    for i in range(len(np.unique(cluster_id))):
        neu_mean[i,:], neu_sem[i,:] = get_mean_sem(
            neu[np.where(cluster_id==i)[0], :].reshape(-1,neu.shape[1]))
    return neu_mean, neu_sem

# compute mean and sem for bined data for clusters.
def get_bin_mean_sem_cluster(bin_neu_seq, cluster_id):
    # get response within cluster at each bin.
    cluster_bin_neu_mean = [get_mean_sem_cluster(neu, cluster_id)[0] for neu in bin_neu_seq]
    cluster_bin_neu_sem  = [get_mean_sem_cluster(neu, cluster_id)[1] for neu in bin_neu_seq]
    # organize into bin_num*n_clusters*time.
    cluster_bin_neu_mean = [np.expand_dims(neu, axis=0) for neu in cluster_bin_neu_mean]
    cluster_bin_neu_sem  = [np.expand_dims(neu, axis=0) for neu in cluster_bin_neu_sem]
    cluster_bin_neu_mean = np.concatenate(cluster_bin_neu_mean, axis=0)
    cluster_bin_neu_sem  = np.concatenate(cluster_bin_neu_sem, axis=0)
    return cluster_bin_neu_mean, cluster_bin_neu_sem
    
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
