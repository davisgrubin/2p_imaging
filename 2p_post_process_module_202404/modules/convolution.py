import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d

# Function to create the right-half Gaussian kernel


def right_half_gaussian_kernel(std_dev, kernel_size):
    """
    Create a right-half Gaussian kernel.

    Parameters:
    std_dev : float
        The standard deviation of the Gaussian kernel.
    kernel_size : int
        The size of the Gaussian kernel.

    Returns:
    right_half_gaussian : numpy array
        The right-half of a Gaussian kernel.
    """
    x = np.linspace(
        0, 5*std_dev, kernel_size)  # Range [0, 3*std_dev] for the right-half
    gaussian_full = np.exp(-x**2 / (2 * std_dev**2))
    gaussian_right_half = gaussian_full / \
        gaussian_full.sum()  # Normalize to ensure sum = 1

    return gaussian_right_half

# Updated denoise function using the right-half Gaussian

# Updated denoise function using the right-half Gaussian


def gaussian_kernel(std_dev, kernel_size):
    """
    Create a full Gaussian kernel.

    Parameters:
    std_dev : float
        The standard deviation of the Gaussian kernel.
    kernel_size : int
        The size of the Gaussian kernel.

    Returns:
    gaussian : numpy array
        The full Gaussian kernel.
    """
    x = np.linspace(-3*std_dev, 3*std_dev,
                    kernel_size)  # Range [-3*std_dev, 3*std_dev] for a full Gaussian
    gaussian_full = np.exp(-x**2 / (2 * std_dev**2))
    gaussian_full /= gaussian_full.sum()  # Normalize to ensure sum = 1
    return gaussian_full


def denoise(data, kernel_size, std_dev, neurons):
    """
    Apply right-half Gaussian convolution to the input data to smooth sharp spikes.

    Parameters:
    data : numpy array
        The input signal data to be smoothed.
    kernel_size : int
        The size of the kernel window.
    std_dev : float
        The standard deviation of the Gaussian kernel.
    neurons : list of ints
        Indices of the neuron signals to be smoothed.

    Returns:
    smoothed_data : numpy array
        The smoothed signal data after Gaussian convolution.
    """
    smoothed_data = np.zeros_like(data)

    # Create the right-half Gaussian kernel
    kernel = right_half_gaussian_kernel(std_dev, kernel_size)
    # kernel = gaussian_kernel(std_dev=std_dev, kernel_size=kernel_size)

    for n in neurons:
        # Apply convolution using the right-half Gaussian kernel
        # convolved_signal = convolve(data[n], kernel, mode='same')

        # convolved_signal = gaussian_filter1d(
        #     data[n], sigma=std_dev, truncate=(kernel_size / (2 * std_dev)))

        convolved_signal = convolve(data[n], kernel, mode='same')
        convolved_signal = np.roll(
            convolved_signal, shift=int(kernel_size // 2))
        # Clip negative values to zero (to prevent negative spikes in smoothed data)
        smoothed_data[n] = np.clip(convolved_signal, 0, None)

    return smoothed_data
