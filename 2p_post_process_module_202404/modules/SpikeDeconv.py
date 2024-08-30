import numpy as np
import matplotlib.pyplot as plt
from suite2p.extraction import oasis
import h5py
import os

from .convolution import nonneg_conditioned_gaussian, convolve_with_template


def read_raw_voltages(ops):
    # f = h5py.File(
    #     os.path.join(ops['save_path0'], 'raw_voltages.h5'),
    #     'r')
    with h5py.File(os.path.join(ops['save_path0'], 'raw_voltages.h5')) as f:
        vol_time = np.array(f['raw']['vol_time'])
        vol_start = np.array(f['raw']['vol_start'])
        vol_stim_vis = np.array(f['raw']['vol_stim_vis'])
        vol_hifi = np.array(f['raw']['vol_hifi'])
        vol_img = np.array(f['raw']['vol_img'])
        vol_stim_aud = np.array(f['raw']['vol_stim_aud'])
        vol_flir = np.array(f['raw']['vol_flir'])
        vol_pmt = np.array(f['raw']['vol_pmt'])
        vol_led = np.array(f['raw']['vol_led'])

        f.close()
    return [vol_time, vol_start, vol_stim_vis, vol_img,
            vol_hifi, vol_stim_aud, vol_flir,
            vol_pmt, vol_led]


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


def plot_for_neuron(timings, dff, spikes, convolved_spikes, neuron=5, num_deconvs=1):
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
    fig, axs = plt.subplots(3, 1, figsize=(30, 10))
    fig.tight_layout(pad=10.0)

    axs[0].plot(timings, convolved_spikes[neuron, :],
                label='Convolved Spike', color='green')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Convolved DF/F Spikes')
    axs[0].set_title('Convolved DF/F -- Up-Time Plot')
    axs[0].legend()

    axs[1].plot(timings, spikes[neuron, :],
                label='Deconv Spike', color='orange')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Deconv DF/F')
    axs[1].set_title('Deconv DF/F -- Up-Time Plot')
    axs[1].legend()

    axs[2].plot(timings, dff[neuron, :], label='DF/F')
    axs[2].set_xlabel('Time (ms)')
    axs[2].set_ylabel('DF/F')
    axs[2].set_title('DF/F -- Up-Time Plot')
    axs[2].legend()

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
        tau=1.0,
        fs=ops['fs'])
    return spikes


def run(
        ops,
        dff,
        num_deconvs=1,
        oasis_tau=10.0,
        neurons=[5, 10, 100]):

    print('===================================================')
    print('=============== Deconvolving Spikes ===============')
    print('===================================================')

    metrics = read_raw_voltages(ops)
    vol_time = metrics[0]
    vol_img = metrics[3]
    dff = read_dff(ops)

    spikes = dff
    # deconvolve a certain number of times -- pretty much always just one time
    for _ in range(num_deconvs):
        spikes = spike_detect(ops, spikes, tau=oasis_tau)

    uptime, _ = get_trigger_time(vol_time, vol_img)

    print(f'here: {spikes.shape[1]}, {spikes.shape[1] // 25000}')

    print('===================================================')
    print('=============== Convolving Spikes =================')
    print('===================================================')

    # perform convolution with right-half standard Gaussian distribution
    template = nonneg_conditioned_gaussian(
        size=(spikes.shape[1] // 10000), sigma=20)

    convolved_spikes = convolve_with_template(
        signal=spikes, template=template, neurons=neurons)

    # plot for certain of neurons
    for i in neurons:
        plot_for_neuron(timings=uptime, dff=dff, spikes=spikes,
                        convolved_spikes=convolved_spikes, neuron=i, num_deconvs=num_deconvs)

    return convolved_spikes
