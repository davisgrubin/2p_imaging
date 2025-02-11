#!/usr/bin/env python3

import numpy as np
from utils import get_mean_sem
from utils import get_sub_time_idx
from utils import get_ca_transient_multi_sess
from utils import get_roi_label_color
from utils import adjust_layout_neu
from utils import add_legend

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# select example neural traces.
def plot_sess_example_traces(ax, list_labels, list_neural_trials, label_names):
    max_ms = 150000
    n = 20
    r = 0.75
    _, _, c1, _ = get_roi_label_color([-1], 0)
    _, _, c2, _ = get_roi_label_color([1], 0)
    # organize data.
    dff = [nt['dff'] for nt in list_neural_trials] 
    time = [nt['time'] for nt in list_neural_trials]
    min_len = np.nanmin([len(t) for t in time])
    dff = np.concatenate([d[:,:min_len] for d in dff], axis=0)
    time = np.nanmin((np.concatenate([t[:min_len].reshape(1,-1) for t in time], axis=0)),axis=0)
    labels = np.concatenate(list_labels)
    # find time indice.
    def get_time_idx():
        start_time = np.max(time)/5.19961106
        time_img_idx = get_sub_time_idx(time, start_time, start_time+max_ms)
        sub_time_img = time[time_img_idx]
        return time_img_idx, sub_time_img
    # find active neurons with most calcium transients.
    def get_active_neuron(cate, n):
        # correct number.
        idx = np.where(np.in1d(np.concatenate(list_labels), cate))[0]
        n = n if n < len(idx) else len(idx)
        # compute calcium transient.
        list_n_ca, _, _ = get_ca_transient_multi_sess(list_neural_trials)
        # find the best neurons.
        act_idx = np.argsort(np.where(np.in1d(np.concatenate(list_labels), cate), list_n_ca, 0))[-n:]
        return act_idx
    # pick neuron indice.
    def get_idx(dff):
        if len(label_names) == 1:
            cate = int(list(label_names.keys())[-1])
            idx = get_active_neuron([cate], n)
        if len(label_names) == 2:
            idx1 = get_active_neuron([-1], int(r*n))
            idx2 = get_active_neuron([1],  int((1-r)*n))
            idx = np.concatenate([idx1,idx2])
        if len(label_names) == 3:
            idx1 = get_active_neuron([-1], int(r*n))
            idx2 = get_active_neuron([1],  int((1-r)*n))
            idx = np.concatenate([idx1,idx2])
        return idx
    # main.
    time_img_idx, sub_time_img = get_time_idx()
    sub_dff = dff[:, time_img_idx].copy()
    act_idx = get_idx(sub_dff)
    sub_dff = sub_dff[act_idx, :].copy()
    # plot neural traces.
    scale = np.max(np.abs(sub_dff))
    for i in range(len(act_idx)):
        ax.plot(
            sub_time_img, sub_dff[i,:] + i * scale,
            color=[get_roi_label_color([i], 0)[2] for i in labels[act_idx]][i])
    # adjust layout.
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('time (ms)')
    ax.set_title('example traces')
    ax.set_ylim([-0.5*scale,len(act_idx)*scale])
    ax.set_xlim([np.min(sub_time_img), np.max(sub_time_img)])
    add_legend(
        ax,
        [get_roi_label_color([int(k)], 0)[2] for k in label_names.keys()],
        [i[1] for i in label_names.items()], None, None, None,
        'upper right')

# calcium transient analysis.
def plot_ca_transient(axs, list_labels, list_neural_trials, label_names, cate):
    try:
        color0, _, color2, _ = get_roi_label_color(cate, 0)
        label_name = label_names[str(cate[0])]
        # compute calcium transient.
        _, dff_ca_neu, dff_ca_time = get_ca_transient_multi_sess(list_neural_trials)
        # get category.
        dff_ca_cate = np.array(dff_ca_neu,dtype='object')[np.in1d(np.concatenate(list_labels), cate)].copy().tolist()
        # average across trials.
        dff_ca_cate = [get_mean_sem(d)[0].reshape(1,-1) for d in dff_ca_cate]
        dff_ca_cate = np.concatenate(dff_ca_cate, axis=0)
        # find the timing or critical points.
        def get_time_line(m):
            start = np.argmin(m)
            peak = np.argmax(m)
            half = peak + np.where(m[peak:]<np.max(m)/2)[0][0]
            return start, peak, half
        # find the half raise time.
        def get_half_raise_time(m):
            start = np.argmin(m)
            peak = np.argmax(m)
            half = np.where(m[:peak]<np.max(m)/2)[0]
            if len(half) > 0:
                half = dff_ca_time[half[-1]] - dff_ca_time[start]
            else:
                half = np.nan
            return half
        # find the half decay time.
        def get_half_decay_time(m):
            peak = np.argmax(m)
            half = np.where(m[peak:]<np.max(m)/2)[0]
            if len(half) > 0:
                half = dff_ca_time[peak+half[0]] - dff_ca_time[peak]
            else:
                half = np.nan
            return half
        # calcium transient grand average.
        def plot_mean_ca_tran(ax):
            m,s = get_mean_sem(dff_ca_cate)
            # compute bounds.
            upper = np.nanmax(m) + np.nanmax(s)
            lower = np.nanmin(m) - np.nanmax(s)
            # mark times.
            start, peak, half = get_time_line(m)
            ax.axvline(dff_ca_time[start], color=color0, lw=1, linestyle='--')
            ax.axvline(dff_ca_time[peak],  color=color0, lw=1, linestyle='--')
            ax.axvline(dff_ca_time[half],  color=color0, lw=1, linestyle='--')
            # plot results.
            ax.plot(dff_ca_time, m, color=color2)
            ax.fill_between(dff_ca_time, m - s, m + s, color=color2, alpha=0.2)
            # adjust layout.
            adjust_layout_neu(ax)
            ax.set_xlim([np.min(dff_ca_time), np.max(dff_ca_time)])
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.set_xlabel('time since calcium transient triggered (ms)')
            ax.set_title(f'calcium transient average \n {label_name}')
        plot_mean_ca_tran(axs[0])
        # half raise time distribution.
        def plot_half_raise_dist(ax):
            t_range = [0,4000]
            # compute half raise time.
            half = np.apply_along_axis(get_half_raise_time, 1, dff_ca_cate)
            # plot distribution.
            ax.hist(
                half, bins=200, range=t_range,
                weights=np.ones(len(half))/len(half),
                align='mid', color=color2)
            ax.tick_params(axis='y', tick1On=False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('half raise time (ms)')
            ax.set_ylabel('fraction')
            ax.set_xlim(t_range)
            ax.set_title(f'half raise time distribution \n {label_name}')
        plot_half_raise_dist(axs[1])
        # half decay time distribution.
        def plot_half_decay_dist(ax):
            t_range = [0,4000]
            # compute half decay time.
            half = np.apply_along_axis(get_half_decay_time, 1, dff_ca_cate)
            # plot distribution.
            ax.hist(
                half, bins=200, range=t_range,
                weights=np.ones(len(half))/len(half),
                align='mid', color=color2)
            ax.tick_params(axis='y', tick1On=False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('half decay time (ms)')
            ax.set_ylabel('fraction')
            ax.set_xlim(t_range)
            ax.set_title(f'half decay time distribution \n {label_name}')
        plot_half_decay_dist(axs[2])
    except: pass
