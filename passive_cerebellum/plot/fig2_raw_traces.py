#!/usr/bin/env python3

import numpy as np
import matplotlib.ticker as mtick

from utils import norm01
from utils import get_frame_idx_from_time

# select example neural traces.
def plot_sess_example_traces(ax, list_labels, list_neural_trials, label_names):
    max_ms = 150000
    max_n = 50
    color = 'black'
    try:
        dff = [nt['dff'] for nt in list_neural_trials] 
        time = [nt['time'] for nt in list_neural_trials][0]
        # find time indice.
        start_time = np.max(time)/5.19961106
        l_idx, r_idx = get_frame_idx_from_time(time, 0, start_time, start_time+max_ms)
        min_len = r_idx-l_idx
        # get subset data.
        sub_dff = np.concatenate([d[:,l_idx:l_idx+min_len] for d in dff], axis=0)
        sub_time_img = time[l_idx:r_idx]
        labels = np.concatenate(list_labels)
        # define layouts.
        ax.axis('off')
        axs = [ax.inset_axes([ci/len(label_names), 0, 0.8/len(label_names), 1], transform=ax.transAxes)
               for ci in range(len(label_names))]
        # find neurons in category.
        for ai, (ci, cn) in enumerate(label_names.items()):
            idx = np.in1d(labels, int(ci))
            if np.sum(idx) > 0:
                cate_dff = sub_dff[idx,:]
                # correct max number to plot.
                n = max_n if max_n<cate_dff.shape[0] else cate_dff.shape[0]
                cate_dff = cate_dff[np.random.choice(cate_dff.shape[0], n, replace=False),:]
                # plot results.
                for ni in range(cate_dff.shape[0]):
                    axs[ai].plot(sub_time_img, norm01(cate_dff[ni,:])*0.8 + ni, color)
                # adjust layout.
                axs[ai].set_title(cn)
                axs[ai].set_xlim([np.min(sub_time_img), np.max(sub_time_img)])
                axs[ai].set_ylim([-1,cate_dff.shape[0]])
            axs[ai].tick_params(axis='y', tick1On=False)
            axs[ai].spines['left'].set_visible(False)
            axs[ai].spines['right'].set_visible(False)
            axs[ai].spines['top'].set_visible(False)
            axs[ai].set_yticks([])
            axs[ai].set_xlabel('time (ms)')
            axs[ai].xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
    except Exception as e: print(e)
