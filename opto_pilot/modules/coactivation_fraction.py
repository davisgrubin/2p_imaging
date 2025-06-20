# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:40:56 2025

@author: saminnaji3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def event(dff):
    events = []
    for i in range(dff.shape[0]):
        threshold = np.max(np.diff(dff[i]))*0.4
        loc = np.where(dff[i] > threshold)[0]
        events.append(loc)
        return events
        
def run(axs , dff):
    events = event(dff)
    axs.eventplot(events)
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_title('events')
    axs.set_xlabel('time')
    axs.set_ylabel('ROI')
    