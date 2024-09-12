import numpy as np
import matplotlib.pyplot as plt
from suite2p.extraction.dcnv import oasis
import h5py
import os
from scipy.optimize import curve_fit

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
    axs[0].plot(timings[shift:], 0.5 * dff[neuron, :], label='DF/F', alpha=0.5)
    axs[0].plot(timings[shift:], spikes[neuron, :],
                label='Inferred Spike', color='orange')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Inferred Spikes')
    axs[0].set_title('Inferred Spikes -- Up-Time Plot')
    axs[0].legend()

    dff_mean = np.mean(dff[neuron, :])
    smooth_mean = np.mean(convolved_spikes[neuron, :])

    scale = 1 / (4 * smooth_mean)

    axs[1].plot(timings[shift:], dff[neuron, :], label='DF/F', alpha=0.5)
    axs[1].plot(timings[shift:], scale * convolved_spikes[neuron, :],
                label='Convolved Spike', color='red', lw=3)
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('DF/F')
    axs[1].set_title('DF/F & Smoothed Inferred -- Up-Time Plot')
    axs[1].legend()

    # plt.rcParams['savefig.dpi'] = 1000
    plt.show()
    # plt.savefig(f'neuron_{neuron}_plot.pdf')

    # print(len(x), spikes.shape)


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def exponential(x, a):
    return np.exp(-a * x)


def spike_detect(ops, dff, tau=1.25):
    spikes = oasis(
        F=dff,
        batch_size=ops['batch_size'],
        tau=tau,
        fs=ops['fs'])

    # window_size = 51  # Adjust as needed
    # half_window = window_size // 2
    # all_taus = []

    # for neuron in range(spikes.shape[0]):
    #     spike_times = np.where(spikes[neuron] > 0)[0]
    #     for spike_time in spike_times:
    #         if spike_time - half_window >= 0 and spike_time + half_window < spikes.shape[1]:
    #             x = np.arange(window_size)
    #             y = dff[neuron, spike_time - half_window: spike_time + half_window + 1]
    #             y = y - np.min(y)  # Shift to start from 0
    #             try:
    #                 popt, _ = curve_fit(exponential, x, y, p0=[tau], bounds=(0, np.inf))
    #                 all_taus.append(1 / popt[0])  # Convert rate to time constant
    #             except:
    #                 pass  # Skip if curve_fit fails

    # # Default to tau if no valid fits
    # avg_tau = np.mean(all_taus) if all_taus else tau
    # print(f'Average tau: {avg_tau}')
    avg_tau = np.exp(-tau)
    return spikes, avg_tau


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

    spikes, avg_sigma = spike_detect(ops, dff, tau=oasis_tau)
    uptime, _ = get_trigger_time(vol_time, vol_img)

    # smoothing
    smoothed = denoise(spikes, kernel_size=350, std_dev=np.exp(-10 / oasis_tau), neurons=neurons)

    baseline = np.zeros_like(spikes)

    # plot for certain neurons
    for i in neurons:
        plot_for_neuron(timings=uptime, dff=dff, spikes=spikes, baseline=baseline,
                        convolved_spikes=smoothed, neuron=i)

    return smoothed, spikes
