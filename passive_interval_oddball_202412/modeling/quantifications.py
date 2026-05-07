#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm

import torch
from torch import nn

from modeling.utils import norm01

# model.
def trf_model(t, b, a, m, r, s, ramp_sign):
    t_eff = ramp_sign * t.unsqueeze(0)
    r_col = r.unsqueeze(1)
    s_col = s.unsqueeze(1)
    m_col = m.unsqueeze(1)
    a_col = a.unsqueeze(1)
    b_col = b.unsqueeze(1)
    term_factor = a_col / 2
    term_exp = torch.exp((2 * m_col + r_col**2) / (2 * s_col) - t_eff / s_col)
    term_erfc = torch.erfc(
        (m_col + r_col**2 / s_col - t_eff) /
        (torch.sqrt(torch.tensor(2.0, device=t.device)) * r_col)
    )
    return b_col + term_factor * term_exp * term_erfc

def trf_model_pre(t, b, a, m, r, s):
    return trf_model(t, b, a, m, r, s, ramp_sign=-1)

def trf_model_post(t, b, a, m, r, s):
    return trf_model(t, b, a, m, r, s, ramp_sign=1)

# bounded reparameterization:
# latent z in (-inf, inf) -> param p in [lo, hi]
# z = 0 maps to midpoint of [lo, hi]
def latent_to_bounded(z, bounds_lo, bounds_hi):
    span = bounds_hi - bounds_lo
    p = bounds_lo + span * torch.sigmoid(z)
    # handle fixed parameters where lo == hi
    fixed = span == 0
    if fixed.any():
        p = torch.where(fixed.unsqueeze(0), bounds_lo.unsqueeze(0), p)
    return p

# loss function.
class LossFunction(nn.Module):
    def __init__(self, l2_weights=None):
        super().__init__()
        self.loss = nn.HuberLoss(reduction="none", delta=0.05)
        self.l2_weights = nn.Parameter(
            torch.tensor(
                l2_weights if l2_weights is not None else [0, 0, 0, 0, 0],
                dtype=torch.float32
            ),
            requires_grad=False
        )

    def forward(self, pred, target, latent_params):
        mse_per_neuron = self.loss(pred, target).mean(dim=1)
        loss_mse = mse_per_neuron.mean()
        l2_loss = torch.sum(self.l2_weights * torch.mean(latent_params**2, dim=0))
        return loss_mse + l2_loss, mse_per_neuron

# optimization.
def optimize_params(
        x, y, model_fn, n_neurons,
        bounds, max_iter, lr, l2_weights, n_inits, device):
    bounds_lo, bounds_hi = bounds
    best_params = torch.zeros((n_neurons, 5), device=device)
    best_loss = torch.full((n_neurons,), np.inf, device=device)
    criterion = LossFunction(l2_weights).to(device)

    # random init in latent space, centered near 0
    for _ in tqdm(range(n_inits)):
        latent_params = (0.1 * torch.randn((n_neurons, 5), device=device)).requires_grad_(True)
        optimizer = torch.optim.Adam([latent_params], lr=lr)

        # optimize step.
        for _ in range(max_iter):
            optimizer.zero_grad()
            params = latent_to_bounded(latent_params, bounds_lo, bounds_hi)
            pred = model_fn(x, *params.T)
            loss, _ = criterion(pred, y, latent_params)
            loss.backward()
            optimizer.step()

        # pick best results.
        with torch.no_grad():
            params = latent_to_bounded(latent_params, bounds_lo, bounds_hi)
            pred = model_fn(x, *params.T)
            mse = ((pred - y) ** 2).mean(dim=1)
            improved = mse < best_loss
            best_loss[improved] = mse[improved]
            best_params[improved] = params[improved]

    return best_params.detach()

# preprocessing.
def preprocess_data(neu_seq_l, neu_time_l, neu_seq_r, neu_time_r, device):
    # normalize data points.
    nsl = torch.tensor([norm01(x) for x in neu_seq_l], dtype=torch.float32, device=device)
    nsr = torch.tensor([norm01(x) for x in neu_seq_r], dtype=torch.float32, device=device)
    # normalize time.
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
    l2_weights = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # normalization.
    nsl, ntl, nsr, ntr = preprocess_data(neu_seq_l, neu_time_l, neu_seq_r, neu_time_r, device)
    # fit models.
    print("Fitting pre ramp model")
    bounds = [
        torch.tensor([-1, 1, -0.5, 1e-2, 1e-3], dtype=torch.float32, device=device),
        torch.tensor([ 1, 1, 0,    1.0, 1.5], dtype=torch.float32, device=device)]
    params_pre = optimize_params(
        ntl, nsl, trf_model_pre, nsl.shape[0],
        bounds=bounds, max_iter=max_iter, lr=lr,
        l2_weights=l2_weights, n_inits=n_inits, device=device)
    pred_pre, r2_pre = reproduce_and_score(ntl, nsl, trf_model_pre, params_pre)
    trf_param_pre, pred_pre, r2_pre = convert_results(params_pre, pred_pre, r2_pre)
    print("Fitting post ramp model")
    bounds = [
        torch.tensor([-1, 1, 0,    1e-2, 1e-3], dtype=torch.float32, device=device),
        torch.tensor([ 1, 1, 0.02, 0.1,  1], dtype=torch.float32, device=device)]
    params_post = optimize_params(
        ntr, nsr, trf_model_post, nsr.shape[0],
        bounds=bounds, max_iter=max_iter, lr=lr,
        l2_weights=l2_weights, n_inits=n_inits, device=device)
    # evaluate.
    pred_post, r2_post = reproduce_and_score(ntr, nsr, trf_model_post, params_post)
    # convert to numpy.
    trf_param_post, pred_post, r2_post = convert_results(params_post, pred_post, r2_post)
    return [trf_param_pre, pred_pre, r2_pre,
            trf_param_post, pred_post, r2_post]


'''
r2_thres = 0.4
r2_gap = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nsl = torch.tensor([norm01(x) for x in neu_seq_l], dtype=torch.float32, device=device)
nsr = torch.tensor([norm01(x) for x in neu_seq_r], dtype=torch.float32, device=device)
ntl = torch.tensor(norm01(neu_time_l) - 1, dtype=torch.float32, device=device)
ntr = torch.tensor(norm01(neu_time_r), dtype=torch.float32, device=device)

i = (r2_pre > r2_thres) * (r2_pre > r2_post)
fig, axs = plt.subplots(6, 8, figsize=(24, 18))
axs = [x for xs in axs for x in xs]
for ni in range(48):
    axs[ni].scatter(ntl,nsl[i][ni],s=3, color='black')
    axs[ni].plot(ntl,pred_pre[i][ni], color='mediumseagreen')
    axs[ni].spines['right'].set_visible(False)
    axs[ni].spines['top'].set_visible(False)
    axs[ni].set_xlim([-1,1])
    axs[ni].set_yticks([0,1])
    axs[ni].set_xticks([-1,0])
    add_legend(
        axs[ni],
        ['mediumseagreen']*6,
        [rf'$R^2_-={r2_pre[i][ni]:.2f}$',
         rf'$b_-={trf_param_pre[i][ni,0]:.2f}$',
         rf'$m_-={trf_param_pre[i][ni,2]:.2f}$',
         rf'$r_-={trf_param_pre[i][ni,3]:.2f}$',
         rf'$\tau_-={trf_param_pre[i][ni,4]:.2f}$'],
        None, None, None, 'upper right')

i = (r2_post > r2_thres) * (r2_post > r2_pre)
fig, axs = plt.subplots(6, 8, figsize=(24, 18))
axs = [x for xs in axs for x in xs]
for ni in range(48):
    axs[ni].scatter(ntr-1,nsr[i][ni],s=3, color='black')
    axs[ni].plot(ntr-1,pred_post[i][ni], color='coral')
    axs[ni].spines['right'].set_visible(False)
    axs[ni].spines['top'].set_visible(False)
    axs[ni].set_xlim([-1,1])
    axs[ni].set_yticks([0,1])
    axs[ni].set_xticks([-1,0])
    axs[ni].set_xticklabels([0,1])
    add_legend(
        axs[ni],
        ['coral']*6,
        [rf'$R^2_+={r2_post[i][ni]:.2f}$',
         rf'$b_+={trf_param_post[i][ni,0]:.2f}$',
         rf'$m_+={trf_param_post[i][ni,2]:.2f}$',
         rf'$r_+={trf_param_post[i][ni,3]:.2f}$',
         rf'$\tau_+={trf_param_post[i][ni,4]:.2f}$'],
        None, None, None, 'upper right')
'''