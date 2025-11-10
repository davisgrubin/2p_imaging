#!/usr/bin/env python3

import numpy as np
import rastermap as rm
from sklearn.cluster import KMeans
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from modeling.utils import norm01
from modeling.utils import get_mean_sem


# clustering neural response.
def clustering_neu_response_mode(x_in, n_clusters, method="kmeans"):
    print("Running clustering")
    if method == "kshape":
        # feature normalization.
        x_in = TimeSeriesScalerMeanVariance().fit_transform(x_in)
        # run clustering model.
        model = KShape(n_clusters=n_clusters, n_init=1, verbose=True)
        cluster_id = model.fit_predict(x_in)
    if method == "kmeans":
        model = KMeans(init="k-means++", n_clusters=n_clusters, n_init=25)
        cluster_id = model.fit_predict(x_in)
    return cluster_id


# organize cluster labels based on stimulus evoked magnitude.
def remap_cluster_id(kernel_all, n_clusters, cluster_id):
    # get response within cluster.
    glm_mean, _ = get_mean_sem_cluster(kernel_all, n_clusters, cluster_id)
    # normalization.
    glm_mean = np.apply_along_axis(norm01, 1, glm_mean)
    # fit model.
    model = rm.Rastermap(n_clusters=2, locality=1)
    model.fit(glm_mean)
    # sort labels.
    mapping = {val: i for i, val in enumerate(model.isort)}
    cluster_id = np.vectorize(mapping.get)(cluster_id)
    return cluster_id


# retrieve cluster id for category.
def get_cluster_cate(cluster_id_all, list_labels, cate):
    idx = np.concatenate(
        [np.in1d(list_labels[i], cate) for i in range(len(list_labels))]
    )
    cluster_id = cluster_id_all[idx]
    neu_labels = np.concatenate(
        [list_labels[i][np.in1d(list_labels[i], cate)] for i in range(len(list_labels))]
    )
    return cluster_id, neu_labels


# compute mean and sem for clusters.
def get_mean_sem_cluster(neu, n_clusters, cluster_id):
    neu_mean = np.zeros((n_clusters, neu.shape[1])) * np.nan
    neu_sem = np.zeros((n_clusters, neu.shape[1])) * np.nan
    for ci in range(n_clusters):
        print(ci)
        idx = np.where(cluster_id == ci)[0]
        if len(idx) > 0:
            neu_mean[ci, :], neu_sem[ci, :] = get_mean_sem(
                neu[np.where(cluster_id == ci)[0], :].reshape(-1, neu.shape[1])
            )
    return neu_mean, neu_sem


# compute mean and sem for bined data for clusters.
def get_bin_mean_sem_cluster(bin_neu_seq, n_clusters, cluster_id):
    # get response within cluster at each bin.
    cluster_bin_neu_mean = [
        get_mean_sem_cluster(neu, n_clusters, cluster_id)[0] for neu in bin_neu_seq
    ]
    cluster_bin_neu_sem = [
        get_mean_sem_cluster(neu, n_clusters, cluster_id)[1] for neu in bin_neu_seq
    ]
    # organize into bin_num*n_clusters*time.
    cluster_bin_neu_mean = [np.expand_dims(neu, axis=0) for neu in cluster_bin_neu_mean]
    cluster_bin_neu_sem = [np.expand_dims(neu, axis=0) for neu in cluster_bin_neu_sem]
    cluster_bin_neu_mean = np.concatenate(cluster_bin_neu_mean, axis=0)
    cluster_bin_neu_sem = np.concatenate(cluster_bin_neu_sem, axis=0)
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
