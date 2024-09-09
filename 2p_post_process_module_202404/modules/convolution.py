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


# Function to perform denoising for each neuron


# Function to perform denoising for each neuron

# def smooth_calcium_data(data, lambd_tv=1.0, lambd_l2=1.0, huber_threshold=0.1):
#     smoothed_data = np.zeros_like(data)

#     for i in range(data.shape[0]):  # Loop over each neuron
#         # Define variables
#         x = cp.Variable(data.shape[1])  # Smoothed signal for each neuron

#         # Huber loss: behaves like L2 for small changes, and like L1 for larger changes (spikes)
#         huber_loss = cp.sum(cp.huber(x - data[i], huber_threshold))

#         # Total variation regularization term to enforce smoothness but allow for sharp spikes
#         tv_penalty = cp.norm1(cp.diff(x))

#         # Define the objective: minimize the Huber loss + lambda * TV penalty
#         objective = cp.Minimize(huber_loss + lambd_tv * tv_penalty)

#         # Solve the optimization problem
#         prob = cp.Problem(objective)
#         prob.solve()

#         # Store the smoothed signal
#         smoothed_data[i] = x.value

#     return smoothed_data


# # Sample data: replace this with your actual calcium imaging data
# num_neurons, sequence_length = 10, 1000
# # Replace with your actual data
# calcium_data = np.random.randn(num_neurons, sequence_length)

# # Denoise the calcium imaging data
# lambd_tv = 0.5  # Adjust to control how much smoothing is applied
# lambd_l2 = 0.5  # Adjust to control how closely the smoothed signal follows the original data
# huber_threshold = 0.8  # Huber loss threshold, tweak this to control sensitivity to spikes
# smoothed_data = smooth_calcium_data(
#     calcium_data, lambd_tv=lambd_tv, lambd_l2=lambd_l2, huber_threshold=huber_threshold)

# # Plotting the original and smoothed signals for comparison
# neuron_index = 0  # Select the neuron you want to visualize
# plt.figure(figsize=(12, 6))
# plt.plot(calcium_data[neuron_index], label="Original Signal")
# plt.plot(smoothed_data[neuron_index], label="Smoothed Signal", linewidth=2)
# plt.legend()
# plt.title(f"Neuron {neuron_index} Calcium Signal: Original vs Smoothed")
# plt.xlabel("Time")
# plt.ylabel("Signal")
# plt.show()
