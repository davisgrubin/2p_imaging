import numpy as np
import matplotlib.pyplot as plt
from suite2p.extraction.dcnv import oasis
import h5py
import os

from .convolution import denoise


def read_raw_voltages(ops):
    # f = h5py.File(
    #     os.path.join(ops['save_path0'], 'raw_voltages.h5'),
    #     'r')
    with h5py.File(os.path.join(ops['save_path0'], 'raw_voltages.h5')) as f:
        vol_time = np.array(f['raw']['vol_time'])
        vol_img = np.array(f['raw']['vol_img'])
        f.close()

    return vol_time, vol_img


def get_trigger_time(
        vol_time,
        vol_bin
):
    # find the edge with np.diff and correct it by preappend one 0.
    diff_vol = np.diff(vol_bin, prepend=0)
    idx_up = np.where(diff_vol == 1)[0]
    idx_down = np.where(diff_vol == -1)[0]
    # select the indice for rising and falling.
    # give the edges in ms.
    time_up = vol_time[idx_up]
    time_down = vol_time[idx_down]
    return time_up, time_down


def read_dff(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'dff.h5'), 'r')
    dff = np.array(f['dff'])
    f.close()
    return dff


def plot_for_neuron(timings, dff, spikes, baseline, convolved_spikes, neuron=5):
    """
    Plots DF/F and deconvolved spike data for a specific neuron.

    Args:
        timings (np.array): Array of time points.
        dff (np.array): DF/F data array.
        spikes (np.array): Spike detection data array.
        neuron (int): Index of the neuron to plot. Default is 5.
        num_deconvs (int): Number of deconvolutions performed. Default is 1.
    """
    plt.figure(figsize=(30, 10))
    fig, axs = plt.subplots(2, 1, figsize=(30, 10))
    fig.tight_layout(pad=10.0)
    shift = len(timings) - len(spikes[neuron, :])
    axs[0].plot(timings[shift:], spikes[neuron, :],
                label='Inferred Spike', color='orange')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Inferred Spikes')
    axs[0].set_title('Inferred Spikes -- Up-Time Plot')
    axs[0].legend()

    axs[1].plot(timings[shift:], dff[neuron, :], label='DF/F', alpha=0.5)
    axs[1].plot(timings[shift:], 1e3 * convolved_spikes[neuron, :],
                label='Convolved Spike', color='red', lw=3)
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('DF/F')
    axs[1].set_title('DF/F & Smoothed Inferred -- Up-Time Plot')
    axs[1].legend()

    plt.savefig(f'neuron_{neuron}_plot.png')

    # print(len(x), spikes.shape)


def spike_detect(
        ops,
        dff,
        tau=1.25
):

    # oasis for spike detection.
    spikes = oasis(
        F=dff,
        batch_size=ops['batch_size'],
        # tau=ops['tau'],
        tau=tau,
        fs=ops['fs'])

    return spikes


def run(
        ops,
        dff,
        oasis_tau=10.0,
        neurons=[5, 10, 100]):

    print('===================================================')
    print('=============== Deconvolving Spikes ===============')
    print('===================================================')

    print('fs: ', ops['fs'])
    metrics = read_raw_voltages(ops)
    vol_time = metrics[0]
    # vol_img = metrics[3]
    vol_img = metrics[1]
    # dff = read_dff(ops)

    spikes = spike_detect(ops, dff, tau=oasis_tau)
    uptime, _ = get_trigger_time(vol_time, vol_img)

    # smoothing
    smoothed = denoise(spikes, neurons=neurons, kernel_size=1500, std_dev=330)
    baseline = np.zeros_like(spikes)

    # plot for certain neurons
    for i in neurons:
        plot_for_neuron(timings=uptime, dff=dff, spikes=spikes, baseline=baseline,
                        convolved_spikes=smoothed, neuron=i)

    return smoothed, spikes
