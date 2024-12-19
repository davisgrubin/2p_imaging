#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
device = torch.device('cuda')
import slicetca

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
