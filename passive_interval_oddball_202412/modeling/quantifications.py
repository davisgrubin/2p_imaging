
import numpy as np
from tqdm import tqdm

import torch
from torch import nn

from modeling.utils import norm01

# model.
def trf_model(t, b, a, m, r, s, ramp_sign=1, sign=1):
    t_eff = ramp_sign * t.unsqueeze(0)
    term_factor = sign * torch.abs(a).unsqueeze(1) / 2
    term_exp = torch.exp((2*m.unsqueeze(1) + r.unsqueeze(1)**2) / (2*s.unsqueeze(1)) - t_eff/s.unsqueeze(1))
    term_erfc = torch.erfc((m.unsqueeze(1) + (r.unsqueeze(1)**2)/s.unsqueeze(1) - t_eff) /
                           (torch.sqrt(torch.tensor(2.0, device=t.device)) * r.unsqueeze(1)))
    return b.unsqueeze(1) + term_factor * term_exp * term_erfc
def trf_model_up(t, b, a, m, r, s):
    return trf_model(t, b, a, m, r, s, ramp_sign=1, sign=1)
def trf_model_dn(t, b, a, m, r, s):
    return trf_model(t, b, a, m, r, s, ramp_sign=-1, sign=1)

# loss function.
class LossFunction(nn.Module):
    def __init__(self, l2_weights=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.l2_weights = nn.Parameter(
            torch.tensor(l2_weights if l2_weights is not None else [0,0,0,0,0],
                         dtype=torch.float32), requires_grad=False)
    def forward(self, pred, target, params):
        mse_per_neuron = self.mse(pred, target).mean(dim=1)
        loss_mse = mse_per_neuron.mean()
        l2_loss = torch.sum(self.l2_weights * torch.mean(params**2, dim=0))
        return loss_mse + l2_loss, mse_per_neuron

# optimization.
def optimize_params(
        x, y, model_fn, n_neurons,
        max_iter, lr, l2_weights, n_inits, device):
    best_params = torch.zeros((n_neurons, 5), device=device)
    best_loss = torch.full((n_neurons,), np.inf, device=device)

    bounds_lo = torch.tensor([-1, 0.0, -0.5, 0.01,    0], dtype=torch.float32, device=device)
    bounds_hi = torch.tensor([ 1, 5.0,    0,   1.0, 5.0], dtype=torch.float32, device=device)
    criterion = LossFunction(l2_weights)

    for _ in tqdm(range(n_inits)):
        # random init
        init = torch.rand((n_neurons, 5), device=device)
        params = (bounds_lo + (bounds_hi - bounds_lo) * init).requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()
            pred = model_fn(x, *params.T)
            loss, _ = criterion(pred, y, params)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                params.data = torch.max(torch.min(params.data, bounds_hi), bounds_lo)

        with torch.no_grad():
            pred = model_fn(x, *params.T)
            mse = ((pred - y) ** 2).mean(dim=1)
            improved = mse < best_loss
            best_loss[improved] = mse[improved]
            best_params[improved] = params[improved]

    return best_params.detach()

# preprocessing.
def preprocess_data(neu_seq_l, neu_time_l, neu_seq_r, neu_time_r, device):
    nsl = torch.tensor([norm01(x) for x in neu_seq_l], dtype=torch.float32, device=device)
    nsr = torch.tensor([norm01(x) for x in neu_seq_r], dtype=torch.float32, device=device)
    ntl = torch.tensor(norm01(neu_time_l) - 1, dtype=torch.float32, device=device)
    ntr = torch.tensor(norm01(neu_time_r), dtype=torch.float32, device=device)
    return nsl, ntl, nsr, ntr

# compute goodness of fit.
def reproduce_and_score(x, y, model_fn, params):
    with torch.no_grad():
        pred = model_fn(x, *params.T)
        ss_res = torch.sum((y - pred) ** 2, dim=1)
        ss_tot = torch.sum((y - y.mean(dim=1, keepdim=True)) ** 2, dim=1)
        r2 = 1 - ss_res / ss_tot
    return pred, r2

# convert results.
def convert_results(params, pred, r2):
    return params.cpu().numpy(), pred.cpu().numpy(), r2.cpu().numpy()

# main.
def fit_trf_model(neu_seq_l, neu_time_l, neu_seq_r, neu_time_r):
    # hyperparameters.
    max_iter = 520
    n_inits = 5
    lr = 1e-2
    l2_weights = 1e-10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # normalization.
    nsl, ntl, nsr, ntr = preprocess_data(neu_seq_l, neu_time_l, neu_seq_r, neu_time_r, device)
    # fit models.
    print("Fitting ramp up model")
    params_up = optimize_params(
        ntl, nsl, trf_model_up, nsl.shape[0],
        max_iter=max_iter, lr=lr, l2_weights=l2_weights, n_inits=n_inits, device=device)
    pred_up, r2_up = reproduce_and_score(ntl, nsl, trf_model_up, params_up)
    trf_param_up, pred_up, r2_up = convert_results(params_up, pred_up, r2_up)
    print("Fitting ramp down model")
    params_dn = optimize_params(
        ntr, nsr, trf_model_dn, nsr.shape[0],
        max_iter=max_iter, lr=lr, l2_weights=l2_weights, n_inits=n_inits, device=device)
    # evaluate.
    pred_dn, r2_dn = reproduce_and_score(ntr, nsr, trf_model_dn, params_dn)
    # convert to numpy.
    trf_param_dn, pred_dn, r2_dn = convert_results(params_dn, pred_dn, r2_dn)
    return [trf_param_up, pred_up, r2_up,
            trf_param_dn, pred_dn, r2_dn]

'''

axs[0].plot(ntl, trf_model_up(ntl, 0, 1.5, -0.4, 0.15, 0.5))
axs[8].plot(ntl, trf_model_up(ntl, 0.2, 1.5, -0.1, 0.1, 0.5))
axs[10].plot(ntl, trf_model_up(ntl, 0.2, 1.5, -0.1, 0.1, 0.5))

fig, axs = plt.subplots(6, 8, figsize=(12, 8))
axs = [x for xs in axs for x in xs]
for ni in range(48):
    axs[ni].plot(ntl,nsl[ni])
    axs[ni].plot(ntl,pred_up[ni])
    axs[ni].axis('off')
    
fig, axs = plt.subplots(6, 8, figsize=(12, 8))
axs = [x for xs in axs for x in xs]
for ni in range(48):
    axs[ni].plot(ntr,nsr[ni])
    axs[ni].plot(ntr,pred_dn[ni])
    axs[ni].axis('off')

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.plot(ntr,nsr[12])
ax.plot(ntr, trf_model_dn(ntr, 0, 1.5, -0.5, 0.4, 1))

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.hist(r2_all_up,bins=100)

thres = 0.251314
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.axis('off')
ax_hm = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
ax_cb = ax.inset_axes([0.6, 0, 0.1, 1], transform=ax.transAxes)
plot_heatmap_neuron(ax_hm, ax_cb, nsl[r2_all_up>thres], ntl, nsl[r2_all_up>thres], norm_mode='minmax')

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.axis('off')
ax_hm = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
ax_cb = ax.inset_axes([0.6, 0, 0.1, 1], transform=ax.transAxes)
plot_heatmap_neuron(ax_hm, ax_cb, nsr[r2_all_dn>thres], ntr, nsr[r2_all_dn>thres], norm_mode='minmax')
'''
            