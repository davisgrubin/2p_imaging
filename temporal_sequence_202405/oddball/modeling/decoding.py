#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
device = torch.device('cuda')

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# single trial decoding by sampling subset of neurons. 
def multi_sess_decoding_num_neu(
        neu_x, neu_y,
        num_step, n_decode,
        ):
    mode = 'temporal'
    n_sess = len(neu_x)
    # define sampling numbers.
    max_num = np.nanmax([neu_x[i].shape[1] for i in range(n_sess)])
    sampling_nums = np.arange(num_step, ((max_num//num_step)+1)*num_step, num_step)
    # run decoding.
    acc_model  = []
    acc_chance = []
    for n_neu in tqdm(sampling_nums):
        results_model = []
        results_chance = []
        for s in range(n_sess):
            # not enough neurons.
            if n_neu > neu_x[s].shape[1]:
                results_model.append(np.nan)
                results_chance.append(np.nan)
            # random sampling n_decode times.
            else:
                for _ in range(n_decode):
                    sub_idx = np.random.choice(neu_x[s].shape[1], n_neu, replace=False)
                    x = neu_x[s][:,sub_idx].copy()
                    y = neu_y[s].copy()
                    am, ac = decoding_spatial_temporal(x, y, mode)
                    results_model.append(am)
                    results_chance.append(ac)
        acc_model.append(np.array(results_model).reshape(-1,1))
        acc_chance.append(np.array(results_chance).reshape(-1,1))
    return sampling_nums, acc_model, acc_chance

# single trial decoding by sliding window.
def multi_sess_decoding_slide_win(
        neu_x, neu_y,
        start_idx, end_idx, win_step,
        n_decode, num_frames,
        ):
    mode = 'temporal'
    n_sess = len(neu_x)
    # run decoding.
    acc_model  = []
    acc_chance = []
    for i in tqdm(range(start_idx, end_idx, win_step)):
        results_model = []
        results_chance = []
        for s in range(n_sess):
            x = neu_x[s][:,:,i-num_frames:i].copy()
            y = neu_y[s].copy()
            am, ac = decoding_spatial_temporal(x, y, mode)
            results_model.append(am)
            results_chance.append(ac)
        acc_model.append(np.array(results_model).reshape(-1,1))
        acc_chance.append(np.array(results_chance).reshape(-1,1))
    return acc_model, acc_chance

# run pytorch spatial-temporal model for single trial decoding.
'''
n_trials, n_neurons, time = 1000, 100, 15
y = np.random.randint(0, 4, n_trials)
x = np.random.rand(n_trials, n_neurons, time)
for i in range(4):
    x[y==i,:,:] += i*10
'''
def decoding_spatial_temporal(x, y, mode):
    test_size = 0.2
    val_size = 0.1
    lr = 1e-2
    batch_size = 64
    epochs = 50
    # split train/val/test sets.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size)
    # build datasets.
    train_loader = DataLoader(
        neu_dataset(x_train, y_train),
        batch_size=batch_size, shuffle=True, drop_last=True)
    x_shuffle = torch.tensor(np.random.permutation(x_test), dtype=torch.float32).to(device)
    y_shuffle = torch.tensor(y_test, dtype=torch.long).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    # define models.
    if mode == 'spatial':
        model = decoder_linear_spatial(
            d_neu  = x_train.shape[1],
            d_time = x_train.shape[2],
            d_out  = len(np.unique(y_train)))
    if mode == 'temporal':
        model = decoder_linear_temporal(
            d_neu  = x_train.shape[1],
            d_time = x_train.shape[2],
            d_out  = len(np.unique(y_train)))
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    objective = nn.NLLLoss(reduction='mean')
    # iteration loop.
    for epoch in range(epochs):
        # training.
        model.train()
        for i, (x_in, y_in) in enumerate(train_loader):
            optimizer.zero_grad()
            y_out = model(x_in)
            loss = objective(y_out, y_in)
            loss.backward()
            optimizer.step()
    # evaluate on test set.
    model.eval()
    with torch.no_grad():
        # model.
        y_pred = model(x_test)
        y_pred = y_pred.argmax(dim=1, keepdim=True)
        acc_test = y_pred.eq(y_test.view_as(y_pred)).sum().item()/len(y_test)
        # chance level.
        y_pred = model(x_shuffle)
        y_pred = y_pred.argmax(dim=1, keepdim=True)
        acc_shuffle = y_pred.eq(y_shuffle.view_as(y_pred)).sum().item()/len(y_shuffle)
    return acc_test, acc_shuffle

# build torch dataset for single trial neural data.
class neu_dataset(Dataset):
    def __init__(
            self,
            x, y,
            transform=None, target_transform=None):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.long).to(device)
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        x = self.x[idx,:,:]
        y = self.y[idx]
        if self.transform:
            x = self.transform(self.x)
        if self.target_transform:
            y = self.target_transform(self.y)
        return x, y
    
# linear decoder compressing neuron axis first.
class decoder_linear_spatial(nn.Module):
    def __init__(self, d_neu, d_time, d_out):
        super(decoder_linear_spatial, self).__init__()
        self.linear1 = nn.Linear(d_neu, 1)
        self.linear2 = nn.Linear(d_time, d_out)
        self.out = nn.LogSoftmax(dim=1)
    def forward(self, x):
        # [batch_size, n_neurons, time].
        x = x.permute(0,2,1)
        # [batch_size, time, n_neurons].
        x = self.linear1(x).squeeze()
        # [batch_size, time].
        x = self.linear2(x).squeeze()
        # [batch_size, n_label].
        x = self.out(x)
        # [batch_size, n_label].
        return x
    
# linear decoder compressing time axis first.
class decoder_linear_temporal(nn.Module):
    def __init__(self, d_neu, d_time, d_out):
        super(decoder_linear_temporal, self).__init__()
        self.linear1 = nn.Linear(d_time, 1)
        self.linear2 = nn.Linear(d_neu, d_out)
        self.out = nn.LogSoftmax(dim=1)
    def forward(self, x):
        # [batch_size, n_neurons, time].
        x = self.linear1(x).squeeze()
        # [batch_size, n_neurons].
        x = self.linear2(x).squeeze()
        # [batch_size, n_label].
        x = self.out(x)
        # [batch_size, n_label].
        return x







