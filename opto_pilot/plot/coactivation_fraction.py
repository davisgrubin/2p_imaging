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
        loc = [j>threshold for j in dff[i]]
        event_temp = np.array(loc, dtype=int)
        event_temp2 = np.diff(event_temp)
        events_temp3 = np.where(event_temp2 > 0)[0]
        events.append(events_temp3)
    return events
        
def run(axs , events):
    #events = event(dff)
    x = np.arange(1)
    # for i in range(len(events)):
    #    print(len(events[i]))
    axs.eventplot(events[0:3])
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_title('events')
    axs.set_xlabel('time')
    axs.set_ylabel('ROI')
    