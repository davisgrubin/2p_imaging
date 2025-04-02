#!/usr/bin/env python3

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn import preprocessing

from modeling.utils import norm01
from modeling.utils import get_mean_sem
from modeling.utils import get_frame_idx_from_time

# clustering neural response.
def clustering_neu_response_mode(x_in, n_clusters, max_clusters):
    # feature normalization.
    x_in -= np.nanmean(x_in, axis=1, keepdims=True)
    # cluster number must be less than sample number.
    n_clusters = n_clusters if n_clusters < x_in.shape[0] else x_in.shape[0]
    # compute evaluation metrics.
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

# feature extraction and categorization.
def feature_categorization(neu_seq_mean, neu_seq_sem, neu_time):
    stim_onset_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
    def find_ramping_onset(nsm, neu_time):
        window_size = 3
        max_attempts = 4
        threshold_factor = 0.2
        pre_window, post_window = 900, 800
        mask = (neu_time >= -pre_window) & (neu_time <= post_window)
        w_response, w_time = nsm[mask], neu_time[mask]
        zero_idx = np.argmin(np.abs(w_time))
        deriv = np.gradient(w_response, w_time)
        baseline = deriv[w_time < 0]
        std_baseline = np.std(baseline)
        threshold = abs(np.mean(baseline)) + std_baseline / 3
        for _ in range(max_attempts):
            onset_idx = np.where(deriv > threshold)[0]
            if onset_idx.size:
                groups = np.split(onset_idx, np.where(np.diff(onset_idx) != 1)[0] + 1)
                groups = [g for g in groups if len(g) >= window_size]
                if groups:
                    max_len = max(len(g) for g in groups)
                    best = max(
                        (
                            (0.2 / (1 + np.min(np.abs(g - zero_idx))) +
                             0.4 * (np.mean(deriv[g]) / np.max(deriv)) +
                             0.4 * (len(g) / max_len), g[0])
                            for g in groups
                        ),
                        key=lambda x: x[0]
                    )[1]
                    return np.flatnonzero(mask)[best]
            threshold -= std_baseline * threshold_factor
        return None
    
    def analyze_ramping_quality(trace, onset_idx, peak_idx):
        if onset_idx > peak_idx:
            result = {
                'r_squared': np.nan,
                'relative_deviation': np.nan,
                'amplitude_change': np.nan,
                'is_monotonic': False,
                'slope': np.nan,
                }
            return result
        else:
            seg = trace[onset_idx:peak_idx+1]
            t = np.arange(len(seg))
            slope, intercept = np.polyfit(t, seg, 1)
            line = slope * t + intercept
            ss_tot = np.sum((seg - seg.mean())**2)
            r_squared = 1 - np.sum((seg - line)**2) / ss_tot
            rel_dev = np.mean(np.abs(seg - line)) / (seg.max() - seg.min())
            pre_seg = trace[max(0, onset_idx - len(seg)//2):onset_idx]
            pre_mean = pre_seg.mean() if pre_seg.size else np.nan
            amp_change = (seg.mean() - pre_mean) / pre_mean if pre_seg.size and pre_mean != 0 else np.inf
            ws = max(3, len(seg) // 5)
            roll_means = np.array([np.mean(seg[i:i+ws]) for i in range(len(seg)-ws+1)])
            monotonic = np.all(np.diff(roll_means) >= -roll_means.std() * 0.2)
            result = {
                'r_squared': r_squared,
                'relative_deviation': rel_dev,
                'amplitude_change': amp_change,
                'is_monotonic': monotonic,
                'slope': slope,
                }
            return result
    def analyze_neural_response(nsm, nss, response_onset_idx, fs=30):
        if nsm is None or not len(nsm) or np.all(np.isnan(nsm)) or response_onset_idx is None:
            return None
        pre_idx = int(0.9 * fs)
        post_idx = int(0.7 * fs)
        ext_idx = int(0.8 * fs)
        resp_win = nsm[stim_onset_idx - pre_idx:stim_onset_idx + ext_idx]
        resp_sem = nss[stim_onset_idx - pre_idx:stim_onset_idx + ext_idx]
        latency = (response_onset_idx - stim_onset_idx) / fs
        rel_onset = response_onset_idx - stim_onset_idx + pre_idx
        rel_onset = min(rel_onset, len(resp_win) - 1)
        post_onset = resp_win[rel_onset:]
        full_peak_idx = rel_onset + np.argmax(post_onset)
        full_peak_time = (full_peak_idx - pre_idx) / fs
        full_peak_amp = resp_win[full_peak_idx]
        pre_win = resp_win[:pre_idx]
        pre_peak_idx = np.argmax(pre_win)
        pre_peak_time = (pre_peak_idx - pre_idx) / fs
        pre_peak_amp = pre_win[pre_peak_idx]
        onset_to_peak = resp_win[rel_onset:full_peak_idx + 1]
        rise_time = (full_peak_idx - rel_onset) / fs
        rise_slope = (full_peak_amp - onset_to_peak[0]) / rise_time if len(onset_to_peak) > 1 else 0
        snr = full_peak_amp / np.mean(resp_sem)
        if full_peak_idx < len(resp_win) - 1:
            post_peak = resp_win[full_peak_idx:]
            final_amp = np.mean(post_peak)
            amp_reduction = (full_peak_amp - final_amp) / full_peak_amp
            try:
                t = np.arange(len(post_peak)) / fs
                decay_rate = -np.polyfit(t, post_peak, 1)[0]
            except:
                decay_rate = 0
        else:
            final_amp, amp_reduction, decay_rate = full_peak_amp, 0, 0
        sharp_amp_quality = analyze_ramping_quality(resp_win, rel_onset, full_peak_idx)
        min_idx = min_val = None
        if response_onset_idx > stim_onset_idx:
            stim_resp = nsm[stim_onset_idx:response_onset_idx + 1]
            if len(stim_resp) > 3:
                min_idx = stim_onset_idx + np.argmin(stim_resp)
                min_val = np.min(stim_resp)
                pre_baseline = nsm[max(0, stim_onset_idx - int(0.5 * fs)):stim_onset_idx]
                base_mean, base_std = np.mean(pre_baseline), np.std(pre_baseline)
                inh_depth = base_mean - min_val
                rec_amp = nsm[response_onset_idx] - min_val
                inh_latency = (min_idx - stim_onset_idx) / fs
                rec_latency = (response_onset_idx - min_idx) / fs
        inh_crit = {
            'post_stim_onset': response_onset_idx > stim_onset_idx,
            'valid_latency': 0 <= latency <= 0.4,
            'significant_drop': False,
            'clear_minimum': False,
            'recovery_present': False,
            'moderate_recovery': False,
            'pre_stim_activity': False
        }
        if min_idx is not None:
            pre_baseline = nsm[max(0, stim_onset_idx - int(0.5 * fs)):stim_onset_idx]
            base_mean, base_std = np.mean(pre_baseline), np.std(pre_baseline)
            post_seg = nsm[response_onset_idx: min(len(nsm), response_onset_idx + int(0.3 * fs))]
            inh_crit.update({
                'significant_drop': min_val < base_mean - 2 * base_std,
                'clear_minimum': min_val < np.mean(nsm[stim_onset_idx:min_idx]) - base_std,
                'recovery_present': rec_amp > 0.5 * (base_mean - min_val),
                'moderate_recovery': nsm[response_onset_idx] < base_mean + base_std,
                'pre_stim_activity': np.mean(pre_baseline) > np.mean(post_seg) - base_std
            })
        sharp_crit = {
            'valid_latency': -0.15 <= latency <= 0.2,
            'sharp_rise': rise_slope > np.percentile(resp_win, 75),
            'good_snr': snr > 2,
            'clear_peak': full_peak_amp > np.mean(resp_win) + np.std(resp_win),
            'significant_amplitude_change': sharp_amp_quality['amplitude_change'] > 0.2,
            'consistent_increase': sharp_amp_quality['is_monotonic']
        }
        delayed_crit = {
            'delayed_latency': latency > 0.2,
            'good_snr': snr > 2,
            'clear_peak': full_peak_amp > np.mean(resp_win) + 2 * np.std(resp_win),
            'significant_response': full_peak_amp > np.mean(resp_win[:pre_idx]) + 2 * np.std(resp_win[:pre_idx]),
            'low_pre_stim_activity': np.mean(resp_win[:pre_idx]) < np.mean(resp_win[pre_idx:pre_idx + post_idx])
        }
        valid_win = resp_win[pre_idx:rel_onset]
        kernel = max(1, min(5, len(valid_win)))
        delayed_crit['monotonic_increase'] = (
            np.all(np.diff(np.convolve(valid_win, np.ones(kernel) / kernel, mode='valid')) >= -0.01)
            if valid_win.size and kernel else False
        )
        ramping_quality = analyze_ramping_quality(resp_win, rel_onset, full_peak_idx)
        ramping_crit = {
            'early_onset': latency < -0.15,
            'good_snr': snr > 2,
            'significant_amplitude_change': ramping_quality['amplitude_change'] > 0.25,
            'good_linearity': ramping_quality['r_squared'] > 0.7,
            'low_deviation': ramping_quality['relative_deviation'] < 0.15,
            'consistent_increase': ramping_quality['is_monotonic'],
            'positive_slope': ramping_quality['slope'] > 0
        }
        decay_crit = {
            'rapid_decay': decay_rate > 0.1,
            'significant_reduction': amp_reduction > 0.2,
            'low_sustained': final_amp < 0.8 * full_peak_amp
        }
        sharp_w = {'valid_latency': 0.0, 'sharp_rise': 0.25, 'good_snr': 0.15, 'clear_peak': 0.15,
                   'significant_amplitude_change': 0.15, 'consistent_increase': 0.3}
        inh_w = {'post_stim_onset': 0.05, 'valid_latency': 0.30, 'significant_drop': 0.25, 'clear_minimum': 0.15,
                 'recovery_present': 0.10, 'moderate_recovery': 0.05, 'pre_stim_activity': 0.10}
        delayed_w = {'delayed_latency': 0.25, 'good_snr': 0.1, 'clear_peak': 0.1, 'significant_response': 0.20,
                     'low_pre_stim_activity': 0.2, 'monotonic_increase': 0.15}
        ramping_w = {'early_onset': 0.2, 'good_snr': 0.1, 'significant_amplitude_change': 0.2,
                     'good_linearity': 0.15, 'low_deviation': 0.15, 'consistent_increase': 0.15,
                     'positive_slope': 0.05}
        decay_w = {'rapid_decay': 0.35, 'significant_reduction': 0.35, 'low_sustained': 0.3}
        inh_conf = sum(inh_crit[k] * inh_w[k] for k in inh_crit)
        sharp_conf = sum(sharp_crit[k] * sharp_w[k] for k in sharp_crit)
        delayed_conf = sum(delayed_crit[k] * delayed_w[k] for k in delayed_crit)
        ramping_conf = sum(ramping_crit[k] * ramping_w[k] for k in ramping_crit)
        decay_conf = sum(decay_crit[k] * decay_w[k] for k in decay_crit)
        if ramping_conf > 0.7 and latency < -0.1:
            rtype = 'ramping_pre_stim' if full_peak_time < 0 else 'ramping_during_stim'
            conf_score = ramping_conf
        elif sharp_conf > 0.7 and -0.15 <= latency <= 0.2:
            rtype = 'sharp_transient_stim_driven' if decay_conf > 0.3 else 'sharp_sustained_stim_driven'
            conf_score = sharp_conf
        elif delayed_conf > 0.7:
            rtype, conf_score = 'delayed_evoked_response', delayed_conf
        elif inh_conf > 0.7:
            rtype, conf_score = 'inhibited', inh_conf
        else:
            rtype, conf_score = 'not_defined', max(sharp_conf, delayed_conf, ramping_conf, inh_conf)
        metrics = {
            'response_latency': latency,
            'peak_time': full_peak_time,
            'peak_amplitude': full_peak_amp,
            'pre_stim_peak_time': pre_peak_time,
            'pre_stim_peak_amplitude': pre_peak_amp,
            'rise_slope': rise_slope,
            'decay_rate': decay_rate,
            'final_amplitude': final_amp,
            'amplitude_reduction': amp_reduction,
            'snr': snr,
            'sharp_criteria': sharp_crit,
            'delayed_criteria': delayed_crit,
            'ramping_criteria': ramping_crit,
            'decay_criteria': decay_crit,
            'ramping_quality': ramping_quality,
            'sharp_amplitude_quality': sharp_amp_quality
        }
        if min_idx is not None:
            metrics.update({
                'inhibition_metrics': {
                    'inhibition_depth': inh_depth,
                    'recovery_amplitude': rec_amp,
                    'inhibition_latency': inh_latency,
                    'recovery_latency': rec_latency,
                    'min_value': min_val,
                    'min_time': (min_idx - stim_onset_idx) / fs
                },
                'inhibition_criteria': inh_crit
            })
        metrics.update({
            'response_type': rtype,
            'confidence_score': conf_score,
            })
        return metrics
    def run_decision():
        results_all = []
        for ni in tqdm(range(neu_seq_mean.shape[0])):
            response_onset = find_ramping_onset(
                neu_seq_mean[ni,:], neu_time)
            result = analyze_neural_response(
                neu_seq_mean[ni,:], neu_seq_sem[ni,:],
                response_onset_idx=response_onset)
            results_all.append(result)
        return results_all
    # run categorization from features.
    results_all = run_decision()
    cluster_labels = np.array([m['response_type'] for m in results_all], dtype='object')
    le = preprocessing.LabelEncoder()
    cluster_id = le.fit(cluster_labels).transform(cluster_labels)
    # relabel based on the number of elements.
    unique, counts = np.unique(cluster_id, return_counts=True)
    sorted_labels = unique[np.argsort(-counts)]
    mapping = {val: i for i, val in enumerate(sorted_labels)}
    cluster_id = np.vectorize(mapping.get)(cluster_id)
    # combine results.
    results_all = pd.DataFrame(results_all)
    results_all['cluster_id'] = cluster_id
    return results_all

'''
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
for ci in np.unique(results_all['cluster_id']):
    ax.plot(neu_time, np.mean(neu_seq_mean[results_all['cluster_id']==ci,:],axis=0))
    ax.legend()
ax.vlines([0,200],-0.1,0.1,color='black',linestyle=':')
'''