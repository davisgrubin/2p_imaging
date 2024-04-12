#!/usr/bin/env python3

import os

from postprocess.ReadResults import read_masks
from postprocess.ReadResults import read_raw_voltages
from postprocess.ReadResults import read_dff
from postprocess.ReadResults import read_neural_trials
from postprocess.ReadResults import read_move_offset

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


#%% ppc

from plot.fig1_mask import plot_ppc_func_max
from plot.fig1_mask import plot_ppc_func_masks
from plot.fig1_mask import plot_ppc_anat_mean
from plot.fig1_mask import plot_ppc_anat_label_masks
from plot.fig1_mask import plot_ppc_superimpose
from plot.fig1_mask import plot_ppc_shared_masks
from plot.fig2_align_grating import plot_ppc_exc_mean
from plot.fig2_align_grating import plot_ppc_inh_mean
from plot.fig3_align_omi import plot_ppc_exc_omi_mean
from plot.fig3_align_omi import plot_ppc_inh_omi_mean
from plot.fig3_align_omi import plot_ppc_omi_fix_heatmap
from plot.fig3_align_omi import plot_ppc_omi_jitter_heatmap
from plot.misc import plot_motion_offset_hist
from plot.misc import plot_inh_exc_label_pc
from plot.misc import plot_isi_distribution


[labels,
 masks,
 mean_func, max_func,
 mean_anat] = read_masks(ops)

[vol_time,
 vol_start_bin,
 vol_stim_bin,
 vol_img_bin] = read_raw_voltages(ops)

dff = read_dff(ops)

neural_trials = read_neural_trials(ops)

jitter_flag = np.load(
    os.path.join(ops['save_path0'], 'jitter_flag.npy'),
    allow_pickle=True)

[xoff, yoff] = read_move_offset(ops)

# define grid space for ppc.
fig = plt.figure()
gs = GridSpec(6, 8, figure=fig)
# masks.
ax01 = plt.subplot(gs[0:2, 0:2])
ax02 = plt.subplot(gs[2:4, 0:2])
ax03 = plt.subplot(gs[0:2, 2:4])
ax04 = plt.subplot(gs[2:4, 2:4])
ax05 = plt.subplot(gs[0:2, 4:6])
ax06 = plt.subplot(gs[2:4, 4:6])
# grating.
ax07 = plt.subplot(gs[4, 0:2])
ax08 = plt.subplot(gs[5, 0:2])
# omission trace.
ax09 = plt.subplot(gs[4, 2:4])
ax10 = plt.subplot(gs[5, 2:4])
# omission heatmap.
ax11 = plt.subplot(gs[4, 4])
ax12 = plt.subplot(gs[4, 5])

# offset.
ax13 = plt.subplot(gs[0, 6])
# labels.
ax14 = plt.subplot(gs[0, 7])
# isi distribution.
ax15 = plt.subplot(gs[1, 6])


# plot results.

plot_ppc_func_max(ax01, max_func, masks)
plot_ppc_func_masks(ax02, masks)
plot_ppc_anat_mean(ax03, mean_anat, masks, labels)
plot_ppc_anat_label_masks(ax04, masks, labels)
plot_ppc_superimpose(ax05, max_func, mean_anat, masks, labels)
plot_ppc_shared_masks(ax06, masks, labels)

plot_ppc_exc_mean(ax07,
    vol_stim_bin, vol_time, neural_trials, jitter_flag, labels)
plot_ppc_inh_mean(ax08,
    vol_stim_bin, vol_time, neural_trials, jitter_flag, labels)

plot_ppc_exc_omi_mean(ax09,
    vol_stim_bin, vol_time,neural_trials, jitter_flag, labels)
plot_ppc_inh_omi_mean(ax10,
    vol_stim_bin, vol_time,neural_trials, jitter_flag, labels)

plot_ppc_omi_fix_heatmap(ax11,
    vol_stim_bin, vol_time, neural_trials, jitter_flag, labels)
plot_ppc_omi_jitter_heatmap(ax12,
    vol_stim_bin, vol_time, neural_trials, jitter_flag, labels)

plot_motion_offset_hist(ax13, xoff, yoff)

plot_inh_exc_label_pc(ax14, labels)

plot_isi_distribution(ax15, vol_stim_bin, vol_time)

# save figure.
fig.set_size_inches(32, 24)
fig.savefig(os.path.join(
    ops['save_path0'], 'figures', 'session_report_'+session_name+'.pdf'),
    dpi=300)
plt.close()






#%% main function for trialization

def run(ops):
    print('===============================================')
    print('================ visualization ================')
    print('===============================================')

    dff = read_dff(ops)
    print('Creating subfolders for {} ROIs'.format(dff.shape[0]))
    for i in range(dff.shape[0]):
        os.makedirs(os.path.join(
            ops['save_path0'], 'figures', 'ROI_'+str(i).zfill(3)))

