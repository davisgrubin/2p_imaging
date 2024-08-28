import numpy as np
import matplotlib.pyplot as plt


def nonneg_conditioned_gaussian(size, sigma=1):
    """
    Generate the right half of a Gaussian distribution with mean 0 and 
    specified std.
    """
    x = np.arange(0, size)
    gauss = np.exp(-x**2 / (2 * sigma**2))
    return gauss / np.max(gauss)


def convolve_with_template(signal, template, neurons=np.arange(0, 333)):
    """
    Convolve the input signal with a specified template.
    """
    # Normalize the template to ensure it sums to 1
    if np.sum(template) != 1:
        template = template / np.sum(template)

    # make sure the signals and template have the same
    # len
    # assert signal.shape[1] == template.shape[0]

    # Perform the convolution
    # Perform convolution on each neuron independently
    template = np.ravel(template)
    convolved_signal_2d = np.zeros_like(signal)

    for i in neurons:
        print(f'=============== Convolving with Neuron {i} ===============')
        convolved_signal_2d[i] = np.convolve(
            signal[i], template, mode='same')

    return convolved_signal_2d
