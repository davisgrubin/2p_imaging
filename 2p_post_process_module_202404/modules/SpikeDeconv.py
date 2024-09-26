import numpy as np
import matplotlib.pyplot as plt
from suite2p.extraction.dcnv import oasis
import h5py
import os
from scipy.optimize import curve_fit

from .convolution import denoise


def read_raw_voltages(ops):
    """
    Reads raw voltage data from an HDF5 file.

    This function opens the 'raw_voltages.h5' file located in the directory specified by ops['save_path0'],
    and extracts the voltage time and image data.

    Parameters:
    ops (dict): A dictionary containing operation parameters, including the 'save_path0' key
                which specifies the directory where the HDF5 file is located.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - vol_time (np.array): An array of voltage timestamps.
        - vol_img (np.array): An array of voltage image data.

    Raises:
    FileNotFoundError: If the 'raw_voltages.h5' file is not found in the specified directory.
    KeyError: If the required datasets 'vol_time' or 'vol_img' are not present in the HDF5 file.
    """
    with h5py.File(os.path.join(ops['save_path0'], 'raw_voltages.h5')) as f:
        vol_time = np.array(f['raw']['vol_time'])
        vol_img = np.array(f['raw']['vol_img'])
        f.close()

    return vol_time, vol_img


def get_trigger_time(
        vol_time,
        vol_bin
):
    """
    Finds the trigger times from the voltage data.

    This function calculates the time points when the voltage data transitions from 0 to 1 and from 1 to 0.
    It uses the np.diff function to compute the first difference of the voltage data and then identifies the indices
    where this difference changes from positive to negative (rising edge) and from negative to positive (falling edge).

    Parameters:
    vol_time (np.array): An array of voltage timestamps.
    vol_bin (np.array): An array of binary voltage data.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - time_up (np.array): An array of time points when the voltage transitions from 0 to 1.
        - time_down (np.array): An array of time points when the voltage transitions from 1 to 0.
    """
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
    """
    Reads DF/F data from an HDF5 file.

    This function opens the 'dff.h5' file located in the directory specified by ops['save_path0'],
    and extracts the DF/F data.

    Parameters:
    ops (dict): A dictionary containing operation parameters, including the 'save_path0' key
                which specifies the directory where the HDF5 file is located.

    Returns:
    np.array: An array containing the DF/F data.
    """

    f = h5py.File(os.path.join(ops['save_path0'], 'dff.h5'), 'r')
    dff = np.array(f['dff'])
    f.close()
    return dff


def plot_for_neuron_without_smoothed(timings, dff, spikes, convolved_spikes, neuron=5, tau=1.25):
    """
    Produces a figure with three subplots: (top) de-convolved spike traces, (middle) original DFF,
                                           and (bottom) traces + original overlayed

    Args:
        timings (np.array): Array of time points.
        dff (np.array): DF/F data array.
        spikes (np.array): Spike detection data array.
        neuron (int): Index of the neuron to plot. Default is 5.
        num_deconvs (int): Number of deconvolutions performed. Default is 1.
        convolved_spikes (np.array): Convolved spikes data array.
    """
    plt.figure(figsize=(30, 10))
    fig, axs = plt.subplots(3, 1, figsize=(30, 10))
    fig.tight_layout(pad=10.0)
    shift = len(timings) - len(spikes[neuron, :])

    # first plot is just inferred spikes
    # axs[0].plot(timings[shift:], 0.5 * dff[neuron, :], label='DF/F', alpha=0.5)
    axs[0].plot(timings[shift:], spikes[neuron, :],
                label='Inferred Spike', color='orange')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Inferred Spikes')
    axs[0].set_title(
        f'Inferred Spikes -- Up-Time Plot for Neuron {neuron} with Tau={tau}')
    axs[0].legend()

    # second plot is just original dff
    axs[1].plot(timings[shift:], dff[neuron, :], label='Original DF/F')
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('DF/F')
    axs[1].set_title('DF/F -- Up-time Plot')
    axs[1].legend()

    # third plot shows the two overlayed
    axs[2].plot(timings[shift:], 0.5 * dff[neuron, :], label='DF/F', alpha=0.5)
    axs[2].plot(timings[shift:], spikes[neuron, :],
                label='Inferred Spike', color='orange')

    x = np.median(timings)
    y = np.median(spikes[neuron, :])
    # axs[2].set_xlim(14000, 14600)
    # axs[2].set_ylim(-2, 2)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Inferred Spikes')
    axs[2].set_title(f'Traces + Original')
    axs[2].legend()

    # plt.figure(figsize=(30, 30))
    plt.rcParams['savefig.dpi'] = 1500
    plt.savefig(f'plot_results/neuron_{neuron}__tau_{tau}_plot.pdf')
    # plt.show()


def plot_for_neuron_with_smoothed(timings, dff, spikes, convolved_spikes, neuron=5, tau=1.25):
    """
    Plots DF/F and deconvolved spike data for a specific neuron.

    Args:
        timings (np.array): Array of time points.
        dff (np.array): DF/F data array.
        spikes (np.array): Spike detection data array.
        neuron (int): Index of the neuron to plot. Default is 5.
        num_deconvs (int): Number of deconvolutions performed. Default is 1.
        convolved_spikes (np.array): Convolved spikes data array.
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
    axs[0].set_title(
        f'Inferred Spikes -- Up-Time Plot for Neuron {neuron} with Tau={tau}')
    axs[0].legend()

    dff_mean = np.mean(dff[neuron, :])
    smooth_mean = np.mean(convolved_spikes[neuron, :])

    scale = 1 / (4 * smooth_mean)

    axs[1].plot(timings[shift:], dff[neuron, :], label='DF/F', alpha=0.5)
    axs[1].plot(timings[shift:], scale * convolved_spikes[neuron, :],
                label='Convolved Spike', color='red', lw=3)
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('DF/F')
    axs[1].set_title('DF/F & Smoothed Inferred Spikes -- Up-Time Plot')
    axs[1].legend()

    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(f'neuron_{neuron}__tau_{tau}_plot.pdf')
    plt.show()


def spike_detect(ops, dff, tau=1.25):
    """
    Detects spikes in the DF/F data using the OASIS algorithm.

    Args:
        ops (dict): A dictionary containing operation parameters.
        dff (np.array): DF/F data array.
        tau (float, optional): Tau parameter for the OASIS algorithm. Defaults to 1.25.

    Returns:
        np.array: An array containing the detected spikes.
    """
    spikes = oasis(
        F=dff,
        batch_size=ops['batch_size'],
        tau=tau,
        fs=ops['fs'])

    avg_tau = np.exp(-tau)
    return spikes


def run(
        ops,
        dff,
        oasis_tau=10.0,
        neurons=np.arange(100),
        plotting_neurons=[5, 10, 100]):

    print('===================================================')
    print('=============== Deconvolving Spikes ===============')
    print('===================================================')

    metrics = read_raw_voltages(ops)
    vol_time = metrics[0]
    # vol_img = metrics[3]
    vol_img = metrics[1]
    # dff = read_dff(ops)

    spikes = spike_detect(ops, dff, tau=oasis_tau)
    uptime, _ = get_trigger_time(vol_time, vol_img)

    # smoothing
    smoothed = denoise(spikes, kernel_size=(200),
                       std_dev=np.exp(-20 / oasis_tau), neurons=neurons)

    # plot for certain neurons
    for i in plotting_neurons:
        # plot_for_neuron(timings=uptime, dff=dff, spikes=spikes,
        #                 convolved_spikes=smoothed, neuron=i, tau=oasis_tau)
        plot_for_neuron_without_smoothed(timings=uptime, dff=dff, spikes=spikes,
                                         convolved_spikes=smoothed, neuron=i, tau=oasis_tau)

    return smoothed, spikes
