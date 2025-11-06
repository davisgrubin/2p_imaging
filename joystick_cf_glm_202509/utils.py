#!/usr/bin/env python3

import time
import functools
import tracemalloc
import numpy as np
import rastermap as rm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
from datetime import datetime


#%% general data processing

# monitor memory usage.
def show_resource_usage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f'running {func.__name__}')
        tracemalloc.start()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f'--- current time: {datetime.now().strftime("%Y.%m.%d %H:%M:%S")}')
        print(f'--- time cost: {elapsed:.2f}s')
        print(f'--- current memory: {current/1024/1024:.2f} MB')
        print(f'--- memory peak: {peak/1024/1024:.2f} MB')
        return result
    return wrapper

# rescale voltage recordings.
def rescale(data, upper, lower):
    data = data.copy()
    data = ( data - np.nanmin(data) ) / (np.nanmax(data) - np.nanmin(data))
    data = data * (upper - lower) + lower
    return data

# normalization into [0,1].
def norm01(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-5)

# compute the scale parameters when normalizing data into [0,1].
def get_norm01_params(data):
    x_scale = 1 / (np.nanmax(data) - np.nanmin(data))
    x_offset = - np.nanmin(data) / (np.nanmax(data) - np.nanmin(data))
    x_min = np.nanmin(data)
    x_max = np.nanmax(data)
    return x_scale, x_offset, x_min, x_max

# compute mean and sem for 3d array data.
def get_mean_sem(data, win_baseline=None):
    # compute mean.
    m = np.nanmean(data.reshape(-1, data.shape[-1]), axis=0)
    # compute sem.
    std = np.nanstd(data.reshape(-1, data.shape[-1]), axis=0)
    count = np.nansum(~np.isnan(data.reshape(-1, data.shape[-1])), axis=0)
    s = std / np.sqrt(count)
    return m, s


#%% retreating neural data

# compute indice with givn time window for dF/F.
def get_frame_idx_from_time(timestamps, c_time, l_time, r_time):
    l_idx = np.searchsorted(timestamps, c_time+l_time)
    r_idx = np.searchsorted(timestamps, c_time+r_time)
    return l_idx, r_idx


#%% plotting

# return colors from dim to dark with a base color.
def get_cmap_color(n_colors, base_color=None, cmap=None, return_cmap=False):
    c_margin = 0.05
    if base_color != None:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            None, base_color)
    if cmap != None:
        pass
    colors = cmap(np.linspace(c_margin, 1 - c_margin, n_colors))
    colors = [mcolors.to_hex(color, keep_alpha=True) for color in colors]
    if return_cmap:
        return cmap, colors
    else:
        return colors

# sort each row in a heatmap.
def sort_heatmap_neuron(neu_seq_sort, sort_method):
    win_conv = 9
    n_clusters = 3
    locality = 1
    # smooth values.
    smoothed = np.array(
        [np.convolve(row, np.ones(win_conv)/win_conv, mode='same')
         for row in neu_seq_sort])
    if sort_method == 'rastermap':
        # fit model.
        model = rm.Rastermap(n_clusters=n_clusters, locality=locality)
        model.fit(smoothed)
        # get ordering.
        sorted_idx = model.isort
    if sort_method == 'peak_timing':
        # get peak timing.
        peak_time = np.argmax(smoothed, axis=1).reshape(-1)
        # combine and sort.
        sorted_idx = peak_time.argsort()
    if sort_method == 'shuffle':
        sorted_idx = np.random.permutation(np.arange(smoothed.shape[0]))
    return sorted_idx

# apply colormap.
def apply_colormap(data, norm_mode, data_share):
    hm_cmap = plt.cm.hot
    pct = 1
    # no valid data found.
    if len(data) == 0:
        hm_data = np.zeros((0, data.shape[1], 3))
        hm_norm = None
        return hm_data, hm_norm, hm_cmap
    else:
        # binary heatmap.
        if norm_mode == 'binary':
            hm_norm = mcolors.Normalize(vmin=0, vmax=1)
            cs = get_cmap_color(3, cmap=hm_cmap)
            hm_cmap = mcolors.LinearSegmentedColormap.from_list(None, ['#FCFCFC', cs[1]])
        else:
            # no normalization.
            if norm_mode == 'none':
                hm_norm = mcolors.Normalize(vmin=np.nanpercentile(data, pct), vmax=np.nanpercentile(data, 100-pct))
                data = np.clip(data, np.nanpercentile(data, pct), np.nanpercentile(data, 100-pct))
                data = norm01(data)
            # normalized into [0,1].
            elif norm_mode == 'minmax':
                for i in range(data.shape[0]):
                    data[i,:] = norm01(data[i,:])
                hm_norm = mcolors.Normalize(vmin=0, vmax=1)
            # share the global scale.
            elif norm_mode == 'share':
                hm_norm = mcolors.Normalize(vmin=np.nanpercentile(data_share, pct), vmax=np.nanpercentile(data_share, 100-pct))
                data = np.clip(data, np.nanpercentile(data_share, pct), np.nanpercentile(data_share, 100-pct))
                data = norm01(data)
            # handle errors.
            else:
                raise ValueError('norm_mode can only be [binary, none, minmax, share].')
        hm_cmap.set_bad(color='white')
        hm_data = hm_cmap(data)
        hm_data = hm_data[..., :3]
    return hm_data, hm_norm, hm_cmap

# hide all axis.
def hide_all_axis(ax):
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

# adjust layout for heatmap.
def adjust_layout_heatmap(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# add legend into subplots.
def add_legend(ax, colors, labels, n_trials, n_neurons, n_sessions, loc, dim=2):
    if dim == 2:
        plot_args = [[],[]]
    if dim == 3:
        plot_args = [[],[],[]]
    handles = []
    if colors != None and labels != None:
        handles += [
            ax.plot(*plot_args, lw=0, color=colors[i], label=labels[i])[0]
            for i in range(len(labels))]
    if n_trials != None and n_neurons != None:
        handles += [
            ax.plot(*plot_args, lw=0, color='black', label=r'$n_{trial}=$'+str(n_trials))[0],
            ax.plot(*plot_args, lw=0, color='black', label=r'$n_{neuron}=$'+str(n_neurons))[0],
            ax.plot(*plot_args, lw=0, color='black', label=r'$n_{session}=$'+str(n_sessions))[0]]
    ax.legend(
        loc=loc,
        handles=handles,
        labelcolor='linecolor',
        frameon=False,
        framealpha=0)

# add heatmap colorbar.
def add_heatmap_colorbar(ax, cmap, norm, label, yticklabels=None):
    if ax != None:
        hide_all_axis(ax)
        cax = ax.inset_axes([0.3, 0.1, 0.2, 0.8], transform=ax.transAxes)
        if norm == None:
            norm = mcolors.Normalize(vmin=0, vmax=1)
        cbar = ax.figure.colorbar(
            plt.cm.ScalarMappable(
                cmap=cmap,
                norm=norm),
            cax=cax)
        cbar.outline.set_linewidth(0.25)
        cbar.ax.set_ylabel(label, rotation=90, labelpad=10)
        cbar.ax.tick_params(axis='y')
        cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        cbar.ax.yaxis.set_major_locator(
            mtick.FixedLocator([norm.vmin+0.2*(norm.vmax-norm.vmin),
                                norm.vmax-0.2*(norm.vmax-norm.vmin)]))
        if yticklabels != None:
            cbar.ax.set_yticklabels(yticklabels)

#%% basic plotting utils class

def plot_mean_sem(ax, t, m, s, c, l=None, a=1.0):
    ax.plot(t, m, color=c, label=l, alpha=a)
    ax.fill_between(t, m - s, m + s, color=c, alpha=0.25, edgecolor='none')
    ax.set_xlim([np.min(t), np.max(t)])

def plot_heatmap_neuron(
        ax_hm, ax_cb, neu_seq, neu_time, neu_seq_sort,
        sort_method='rastermap',
        norm_mode=None, neu_seq_share=None,
        cbar_label=None,
        ):
    n_yticks = 2
    max_pixel = 258
    if len(neu_seq) > 0:
        # exclude pure nan row.
        neu_idx = np.where(~np.all(np.isnan(neu_seq), axis=1))[0]
        neu_seq = neu_seq[neu_idx,:].copy()
        neu_seq_sort = neu_seq_sort[neu_idx,:].copy()
        # sort heatmap.
        sorted_idx = sort_heatmap_neuron(neu_seq_sort, sort_method=sort_method)
        data = neu_seq[sorted_idx,:].copy()
        n_neurons = data.shape[0]
        # reduce pixels.
        nbin = int(neu_seq.shape[0] / max_pixel)
        data = rm.utils.bin1d(data, bin_size=nbin, axis=0)
        # compute share scale if give.
        if neu_seq_share != None:
            data_share = np.concatenate(neu_seq_share, axis=0)
        else:
            data_share = np.nan
        # prepare heatmap.
        hm_data, hm_norm, hm_cmap = apply_colormap(data, norm_mode, data_share)
        # plot heatmap.
        ax_hm.imshow(
            hm_data,
            extent=[neu_time[0], neu_time[-1], 1, hm_data.shape[0]],
            interpolation='nearest', aspect='auto')
        # adjust layouts.
        adjust_layout_heatmap(ax_hm)
        ax_hm.set_ylabel('neuron id (sorted)')
        ax_hm.tick_params(axis='y', labelrotation=90)
        ax_hm.set_yticks((((np.arange(n_yticks)+0.5)/n_yticks)*data.shape[0]).astype('int32'))
        ax_hm.set_yticklabels((((np.arange(n_yticks)+0.5)/n_yticks)*n_neurons).astype('int32'))
        # add colorbar.
        add_heatmap_colorbar(ax_cb, hm_cmap, hm_norm, 'dF/F')
