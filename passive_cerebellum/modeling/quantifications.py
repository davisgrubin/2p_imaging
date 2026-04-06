#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm

import torch
from torch import nn

from modeling.utils import norm01

# model.
def trf_model(t, b, a, m, r, s, ramp_sign=1):
    t_eff = ramp_sign * t.unsqueeze(0)
    term_factor = a.unsqueeze(1) / 2
    term_exp = torch.exp((-2*m.unsqueeze(1) + r.unsqueeze(1)**2) / (2*s.unsqueeze(1)) - t_eff/s.unsqueeze(1))
    term_erfc = torch.erfc((-1*m.unsqueeze(1) + (r.unsqueeze(1)**2)/s.unsqueeze(1) - t_eff) /
                           (torch.sqrt(torch.tensor(2.0, device=t.device)) * r.unsqueeze(1)))
    return b.unsqueeze(1) + term_factor * term_exp * term_erfc
def trf_model_pre(t, b, a, m, r, s):
    return trf_model(t, b, a, m, r, s, ramp_sign=1)
def trf_model_post(t, b, a, m, r, s):
    return trf_model(t, b, a, m, r, s, ramp_sign=-1)

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
    bounds_lo = torch.tensor([-1, 0, 0, 0.01, 0], dtype=torch.float32, device=device)
    bounds_hi = torch.tensor([ 1, 5.0, 0.5, 1.0, 5.0], dtype=torch.float32, device=device)
    criterion = LossFunction(l2_weights)
    # random init.
    for _ in tqdm(range(n_inits)):
        init = torch.rand((n_neurons, 5), device=device)
        params = (bounds_lo + (bounds_hi - bounds_lo) * init).requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=lr)
        # optimize step.
        for _ in range(max_iter):
            optimizer.zero_grad()
            pred = model_fn(x, *params.T)
            loss, _ = criterion(pred, y, params)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                params.data = torch.max(torch.min(params.data, bounds_hi), bounds_lo)
        # pick best results.
        with torch.no_grad():
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
    l2_weights = 1e-10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # normalization.
    nsl, ntl, nsr, ntr = preprocess_data(neu_seq_l, neu_time_l, neu_seq_r, neu_time_r, device)
    # fit models.
    print("Fitting pre ramp model")
    params_pre = optimize_params(
        ntl, nsl, trf_model_pre, nsl.shape[0],
        max_iter=max_iter, lr=lr, l2_weights=l2_weights, n_inits=n_inits, device=device)
    pred_pre, r2_pre = reproduce_and_score(ntl, nsl, trf_model_pre, params_pre)
    trf_param_pre, pred_pre, r2_pre = convert_results(params_pre, pred_pre, r2_pre)
    print("Fitting post ramp model")
    params_post = optimize_params(
        ntr, nsr, trf_model_post, nsr.shape[0],
        max_iter=max_iter, lr=lr, l2_weights=l2_weights, n_inits=n_inits, device=device)
    # evaluate.
    pred_post, r2_post = reproduce_and_score(ntr, nsr, trf_model_post, params_post)
    # convert to numpy.
    trf_param_post, pred_post, r2_post = convert_results(params_post, pred_post, r2_post)
    return [trf_param_pre, pred_pre, r2_pre,
            trf_param_post, pred_post, r2_post]

'''

axs[0].plot(ntl, trf_model_pre(ntl, 0, 1.5, -0.4, 0.15, 0.5))
axs[8].plot(ntl, trf_model_pre(ntl, 0.2, 1.5, -0.1, 0.1, 0.5))
axs[10].plot(ntl, trf_model_pre(ntl, 0.2, 1.5, -0.1, 0.1, 0.5))


fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.scatter(ntr, nsr[7],s=3, color='black')
ax.plot(ntr, trf_model_post(ntr,
                          torch.tensor([-0.3]),
                          torch.tensor([5.6]),
                          torch.tensor([0.7]),
                          torch.tensor([0.3]),
                          torch.tensor([0.3])).reshape(-1))

fig, axs = plt.subplots(6, 8, figsize=(12, 8))
axs = [x for xs in axs for x in xs]
for ni in range(48):
    axs[ni].plot(ntl,nsl[ni])
    axs[ni].plot(ntl,pred_pre[ni])
    axs[ni].axis('off')
    axs[ni].set_title(f'{r2_pre[ni]:.2f}')
fig, axs = plt.subplots(6, 8, figsize=(12, 8))
axs = [x for xs in axs for x in xs]
for ni in range(48):
    axs[ni].plot(ntr,nsr[ni])
    axs[ni].plot(ntr,pred_post[ni])
    axs[ni].axis('off')
    axs[ni].set_title(f'{r2_post[ni]:.2f}')

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.scatter(r2_pre, r2_post)

fig, axs = plt.subplots(5, 1, figsize=(5, 10))
for i in range(5):
    axs[i].hist(trf_param_pre[:,i], bins=100)
fig, axs = plt.subplots(5, 1, figsize=(5, 10))
for i in range(5):
    axs[i].hist(trf_param_post[:,i], bins=100)
    
i = (r2_pre>0.8)*(r2_post>0.8)
fig, axs = plt.subplots(6, 8, figsize=(18, 12))
axs = [x for xs in axs for x in xs]
for ni in range(48):
    axs[ni].scatter(ntl,nsl[i][ni],s=3, color='black')
    axs[ni].plot(ntl,pred_pre[i][ni], color='mediumseagreen')
    axs[ni].scatter(ntr,nsr[i][ni],s=3, color='black')
    axs[ni].plot(ntr,pred_post[i][ni], color='coral')
    axs[ni].axvline(0, color='black', lw=1, linestyle='-')
    axs[ni].axis('off')
    axs[ni].set_title(f'up:{r2_pre[i][ni]:.2f},dn:{r2_post[i][ni]:.2f}')

i = (r2_pre<0.4)*(r2_post<0.4)
fig, axs = plt.subplots(6, 8, figsize=(18, 12))
axs = [x for xs in axs for x in xs]
for ni in range(48):
    axs[ni].scatter(ntl,nsl[i][ni],s=3, color='black')
    axs[ni].plot(ntl,pred_pre[i][ni], color='mediumseagreen')
    axs[ni].scatter(ntr,nsr[i][ni],s=3, color='black')
    axs[ni].plot(ntr,pred_post[i][ni], color='coral')
    axs[ni].axvline(0, color='black', lw=1, linestyle='-')
    axs[ni].axis('off')
    axs[ni].set_title(f'up:{r2_pre[i][ni]:.2f},dn:{r2_post[i][ni]:.2f}')
    

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.plot(ntr,nsr[12])
ax.plot(ntr, trf_model_post(ntr, 0, 1.5, -0.5, 0.4, 1))

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.hist(r2_all_pre,bins=100)

thres = 0.251314
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.axis('off')
ax_hm = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
ax_cb = ax.inset_axes([0.6, 0, 0.1, 1], transform=ax.transAxes)
plot_heatmap_neuron(ax_hm, ax_cb, nsl[r2_all_pre>thres], ntl, nsl[r2_all_pre>thres], norm_mode='minmax')

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.axis('off')
ax_hm = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
ax_cb = ax.inset_axes([0.6, 0, 0.1, 1], transform=ax.transAxes)
plot_heatmap_neuron(ax_hm, ax_cb, nsr[r2_all_post>thres], ntr, nsr[r2_all_post>thres], norm_mode='minmax')
'''
            