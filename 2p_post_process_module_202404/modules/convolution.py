# import scipy.sparse
# import scipy as sc
# # from scipy.ndimage import gaussian_filter1d
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import cvxpy as cvx
# import statsmodels.api as sm

# """
# def plot_trace(groundtruth=False):

#     plt.figure(figsize=(20,4))
#     plt.subplot(211)
#     plt.plot(b+c, lw=2, label='denoised')

#     if groundtruth:
#         plt.plot(true_b+true_c, c='r', label='truth', zorder=-11)

#     plt.plot(y, label='data', zorder=-12, c='y')
#     plt.legend(ncol=3, frameon=False, loc=(.02,.85))
#     simpleaxis(plt.gca())
#     plt.subplot(212)
#     plt.plot(s, lw=2, label='deconvolved', c='g')

#     if groundtruth:
#         for k in np.where(true_s)[0]:
#             plt.plot([k,k],[-.1,1], c='r', zorder=-11, clip_on=False)

#     plt.ylim(0,1.3)
#     plt.legend(ncol=3, frameon=False, loc=(.02,.85))
#     simpleaxis(plt.gca())

#     print("Correlation of deconvolved activity  with ground truth ('spikes') : %.4f" % np.corrcoef(s,true_s)[0,1])
#     print("Correlation of denoised fluorescence with ground truth ('calcium'): %.4f" % np.corrcoef(c,true_c)[0,1])
# """

# # b + c gives the convolved "denoised" data
# #   b: baseline value of flourescence data
# #   c: inferred denoised signal at each time bin
# # s gives deconvolved data


# def constrained_foopsi(y, g, sn, b=0, solver='ECOS'):
#     """Solves the noise constrained deconvolution problem using the cvxpy package.

#     Parameters:
#     -----------
#     y : array, shape (T,)
#         Fluorescence trace.
#     g : tuple of float
#         Parameters of the autoregressive model, cardinality equivalent to p.
#     sn : float
#         Estimated noise level.
#     b : float, optional, default 0
#         Baseline.
#     solver: string, optional, default 'ECOS'
#         Solvers to be used. Can be choosen between ECOS, SCS, CVXOPT and GUROBI,
#         if installed.

#     Returns:
#     --------
#     c : array, shape (T,)
#         The inferred denoised fluorescence signal at each time-bin.
#     s : array, shape (T,)
#         Discretized deconvolved neural activity (spikes).
#     b : float
#         Fluorescence baseline value.
#     g : tuple of float
#         Parameters of the AR(2) process that models the fluorescence impulse response.
#     lam: float
#         Optimal Lagrange multiplier for noise constraint
#     """

#     T = y.size
#     # construct deconvolution matrix  (s = G*c)
#     G = sc.sparse.dia_matrix((np.ones((1, T)), [0]), (T, T))
#     for i, gi in enumerate(g):
#         G = G + \
#             sc.sparse.dia_matrix((-gi * np.ones((1, T)), [-1 - i]), (T, T))
#     c = cvx.Variable(T)  # calcium at each time step
#     if b is None:
#         b = cvx.Variable(1)
#     # cvxpy had sometime trouble to find solution for G*c
#     objective = cvx.Minimize(cvx.norm(c, 1))
#     constraints = [G * c >= 0]
#     constraints.append(cvx.sum_squares(b + c - y) <= sn * sn * T)
#     prob = cvx.Problem(objective, constraints)
#     prob.solve(solver=solver)
#     try:
#         b = b.value
#     except:
#         pass
#     try:
#         s = np.squeeze(np.asarray(G * c.value))
#         s[0] = 0  # reflects merely initial calcium concentration
#         c = np.squeeze(np.asarray(c.value))
#     except:
#         s = None
#     return c, s, b, g, prob.constraints[1].dual_value


# def constrained_foopsi_denoise_only(y, g, sn, b=0, solver='ECOS'):
#     """
#     Solves the noise-constrained denoising problem without performing deconvolution.

#     Parameters:
#     -----------
#     y : array, shape (T,)
#         Fluorescence trace.
#     g : tuple of float
#         Parameters of the autoregressive model, cardinality equivalent to p.
#     sn : float
#         Estimated noise level.
#     b : float, optional, default 0
#         Baseline.
#     solver: string, optional, default 'ECOS'
#         Solvers to be used. Can be chosen between ECOS, SCS, CVXOPT, and GUROBI,
#         if installed.

#     Returns:
#     --------
#     c : array, shape (T,)
#         The inferred denoised fluorescence signal at each time-bin.
#     b : float
#         Fluorescence baseline value.
#     """

#     T = y.size
#     # Construct denoising matrix (without deconvolution)
#     G = sc.sparse.dia_matrix((np.ones((1, T)), [0]), (T, T))
#     for i, gi in enumerate(g):
#         G = G + \
#             sc.sparse.dia_matrix((-gi * np.ones((1, T)), [-1 - i]), (T, T))

#     # Set up optimization variables (calcium signal c and baseline b)
#     c = cvx.Variable(T)  # calcium at each time step
#     if b is None:
#         b = cvx.Variable(1)

#     # Define objective to minimize the calcium signal under noise constraints
#     objective = cvx.Minimize(cvx.norm(c, 1))
#     constraints = [G * c >= 0]  # signal is non-negative
#     constraints.append(cvx.sum_squares(b + c - y) <= sn * sn * T)

#     # Solve the problem
#     prob = cvx.Problem(objective, constraints)
#     prob.solve(solver=solver)

#     # Retrieve baseline and denoised signal
#     try:
#         b = b.value
#     except:
#         pass
#     try:
#         c = np.squeeze(np.asarray(c.value))
#     except:
#         c = None

#     return c, b


# def constrained_foopsi_multi_neuron(Y, G, sn, b=None, solver='ECOS', neurons=np.arange(333)):
#     """
#     Solves the noise-constrained deconvolution problem for multiple neurons using the cvxpy package.

#     Parameters:
#     -----------
#     Y : 2D array, shape [num_neurons, T]
#         Fluorescence traces for multiple neurons.
#     G : 2D array, shape [num_neurons, p]
#         AR coefficients for each neuron (p is the order of the AR process).
#     sn : float or 1D array, shape [num_neurons]
#         Estimated noise level for each neuron.
#     b : float or 1D array, optional, default None
#         Baseline values for each neuron. If None, it will be inferred.
#     solver : string, optional, default 'ECOS'
#         Solver to be used.

#     Returns:
#     --------
#     C : 2D array, shape [num_neurons, T]
#         The inferred denoised fluorescence signal for each neuron.
#     S : 2D array, shape [num_neurons, T]
#         Discretized deconvolved neural activity (spikes) for each neuron.
#     B : 1D array, shape [num_neurons]
#         Fluorescence baseline values for each neuron.
#     L : 1D array, shape [num_neurons]
#         Optimal Lagrange multipliers for noise constraint for each neuron.
#     """
#     num_neurons, T = Y.shape
#     C = np.zeros_like(Y)  # Denoised signals for all neurons
#     S = np.zeros_like(Y)  # Deconvolved spikes for all neurons
#     B = np.zeros(num_neurons)  # Baseline values for all neurons
#     L = np.zeros(num_neurons)  # Lagrange multipliers for noise constraint

#     for i in neurons:
#         y = Y[i, :]
#         g = G[i, :]

#         # Construct deconvolution matrix G for neuron i
#         G_matrix = sc.sparse.dia_matrix((np.ones((1, T)), [0]), (T, T))
#         for j, gi in enumerate(g):
#             G_matrix = G_matrix + \
#                 scipy.sparse.dia_matrix(
#                     (-gi * np.ones((1, T)), [-1 - j]), (T, T))

#         # Set up optimization variables
#         c = cvx.Variable(T)  # calcium at each time step
#         if b is None:  # If no baseline provided, optimize for b
#             b_i = cvx.Variable(1)
#         else:
#             b_i = b[i]  # Use the provided baseline for neuron i

#         # Objective and constraints
#         objective = cvx.Minimize(cvx.norm(c, 1))  # L1 norm for sparsity
#         constraints = [G_matrix * c >= 0]  # Non-negativity constraint
#         constraints.append(cvx.sum_squares(b_i + c - y) <=
#                            (sn[i] * 1.5) ** 2 * T)  # Noise constraint

#         # Solve the optimization problem
#         prob = cvx.Problem(objective, constraints)
#         prob.solve(solver=solver)

#         # Extract denoised signal, baseline, and deconvolved spikes
#         try:
#             C[i, :] = np.squeeze(np.asarray(c.value))  # Denoised signal
#             B[i] = b_i.value  # Baseline
#         except:
#             pass

#         # Compute deconvolved spikes (s = G * c)
#         try:
#             S[i, :] = np.squeeze(np.asarray(G_matrix @ c.value))
#             S[i, 0] = 0  # Reflect initial calcium concentration
#         except:
#             S[i, :] = None

#         # Store the optimal Lagrange multiplier for noise constraint
#         try:
#             L[i] = constraints[1].dual_value
#         except:
#             L[i] = None

#     return C, S, B, L


# def fit_ar_to_each_neuron(y, lags=1, neurons=np.arange(333)):
#     """
#     Fit an AR model to each neuron's fluorescence trace.

#     Parameters:
#     -----------
#     y : 2D numpy array, shape [num_neurons, milliseconds]
#         Fluorescence traces for multiple neurons.
#     lags : int, optional, default=1
#         Number of lags for the AR model.

#     Returns:
#     --------
#     ar_params : 2D numpy array, shape [num_neurons, lags]
#         AR model coefficients for each neuron.
#     """
#     num_neurons = y.shape[0]
#     # To store AR coefficients for each neuron
#     ar_params = np.zeros((num_neurons, lags))
#     fits = np.zeros_like(y[:, 1:])

#     for i in neurons:
#         neuron_trace = y[i, :]
#         model = sm.tsa.AutoReg(neuron_trace, lags=lags)
#         fit = model.fit()
#         fits[i] = fit.fittedvalues
#         # Store AR coefficients (ignore intercept)
#         ar_params[i] = fit.params[1:]

#     return ar_params, fits


# def denoise(y, neurons=np.arange(333)):
#     # fit an AR model to obtain AR parameters
#     g, fits = fit_ar_to_each_neuron(y, neurons=neurons)

#     # estimate noise level of raw data
#     residuals = y[:, 1:]
#     residuals[neurons] = residuals[neurons] - \
#         np.array([fitvals for fitvals in fits[neurons]])
#     sn = np.std(residuals, axis=-1)

#     denoised_signal, _, baseline, _ = constrained_foopsi_multi_neuron(
#         Y=y, G=g, sn=sn, neurons=neurons)

#     return denoised_signal, baseline

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
