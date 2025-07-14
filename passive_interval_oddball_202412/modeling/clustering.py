#!/usr/bin/env python3

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from modeling.quantifications import run_quantification
from modeling.utils import get_mean_sem

# clustering neural response.
def clustering_neu_response_mode(x_in, n_clusters):
    print('Running clustering')
    # feature normalization.
    x_in -= np.nanmean(x_in, axis=1, keepdims=True)
    # cluster number must be less than sample number.
    n_clusters = n_clusters if n_clusters < x_in.shape[0] else x_in.shape[0]
    # run clustering model.
    model = AgglomerativeClustering(n_clusters)
    cluster_id = model.fit_predict(x_in)
    return cluster_id

# organize cluster labels based on stimulus evoked latency.
def remap_cluster_id(neu, neu_time, n_clusters, cluster_id):
    neu_seq, _ = get_mean_sem_cluster(neu, n_clusters, cluster_id)
    evoke_time = run_quantification(neu_seq, neu_time, win_eval_c=0, samping_size=0)['evoke_latency']
    sorted_labels = np.argsort(evoke_time)[::-1]
    mapping = {val: i for i, val in enumerate(sorted_labels)}
    cluster_id = np.vectorize(mapping.get)(cluster_id)
    return cluster_id

# retrieve cluster id for category.
def get_cluster_cate(cluster_id_all, list_labels, list_significance, cate):
    idx = np.concatenate([np.in1d(list_labels[i],cate)*list_significance[i]['r_standard']
           for i in range(len(list_labels))])
    cluster_id = cluster_id_all[idx]
    neu_labels = np.concatenate([
        list_labels[i][np.in1d(list_labels[i],cate)*list_significance[i]['r_standard']]
        for i in range(len(list_labels))])
    return cluster_id, neu_labels

# compute mean and sem for clusters.
def get_mean_sem_cluster(neu, n_clusters, cluster_id):
    neu_mean = np.zeros((n_clusters, neu.shape[1])) * np.nan
    neu_sem  = np.zeros((n_clusters, neu.shape[1])) * np.nan
    for ci in range(n_clusters):
        idx = np.where(cluster_id==ci)[0]
        if len(idx) > 0:
            neu_mean[ci,:], neu_sem[ci,:] = get_mean_sem(
                neu[np.where(cluster_id==ci)[0], :].reshape(-1,neu.shape[1]))
    return neu_mean, neu_sem

# compute mean and sem for bined data for clusters.
def get_bin_mean_sem_cluster(bin_neu_seq, n_clusters, cluster_id):
    # get response within cluster at each bin.
    cluster_bin_neu_mean = [get_mean_sem_cluster(neu, n_clusters, cluster_id)[0] for neu in bin_neu_seq]
    cluster_bin_neu_sem  = [get_mean_sem_cluster(neu, n_clusters, cluster_id)[1] for neu in bin_neu_seq]
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
    neu_mean, _ = get_mean_sem_cluster(neu, n_clusters, cluster_id)
    cluster_corr = np.corrcoef(neu_mean)
    return cluster_corr
