# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 17:22:26 2026

@author: saminnaji3
"""

import os
import run_manual_postprocess
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules.ReadResults import read_dff
from modules.ReadResults import save_lables

need_plot = 0

data_date ='20260111'
subject_name = 'SA18'
subject = subject_name + '_LG'

session_data_path = 'C:\\Users\\saminnaji3\\Downloads\\' + subject + '\\' + subject_name + '_' + data_date
output_dir_onedrive = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/2p imaging/Figures/' + subject + '/RawTraces/Good_ROIs/'


ops = np.load(
    os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
    allow_pickle=True).item()
ops['save_path0'] = os.path.join(session_data_path)
dff = read_dff(ops, 'qc_results')

# if you want to mark good ROI write ids in good_roi else mark bad rois in bad_roi
good_roi = []
bad_roi = [578, 577, 576, 575, 574, 573, 572, 571, 569, 568, 567, 566, 565, 564, 563, 562, 561, 560, 559, 558, 557, 556, 555, 554, 553, 552, 551, 550, 548, 547, 546, 545, 
           544, 543, 542, 541, 540, 539, 538, 537, 536, 535, 534, 533, 532, 531, 530, 528, 527, 526, 524, 523, 522, 520, 519, 518, 517, 516, 515, 513, 512, 511, 510, 509, 
           508, 507, 506, 505, 504, 503, 502, 501, 500, 499, 498, 497, 496, 495, 494, 493, 492, 491, 490, 489, 488, 487, 486, 485, 484, 482, 481, 480, 478, 477, 476, 475, 474, 473, 472, 
           471, 469, 468, 466, 465, 463, 462, 460, 459, 458, 457, 456, 455, 454, 453, 452, 451, 450, 449, 448, 447, 446, 445, 443, 441, 440, 439, 438, 437, 436, 435, 434, 433, 431, 430, 
           429, 428, 426, 425, 424, 423, 422, 421, 420, 419, 418, 417, 416, 415, 413, 412, 411, 410, 409, 408, 407, 406, 404, 403, 401, 400, 399, 397, 395, 394, 393, 392, 391, 390, 389, 
           388, 386, 385, 384, 383, 382, 381, 380, 379, 377, 376, 375, 374, 373, 371, 370, 369, 367, 365, 364, 363, 362, 361, 360, 358, 357, 356, 355, 354, 353, 352, 351, 350, 349, 348, 347, 
           346, 344, 343, 342, 341, 339, 338, 337, 336, 335, 333, 331, 330, 328, 326, 325, 324, 323, 322, 321, 320, 318, 315, 314, 312, 311, 310, 309, 308, 305, 303, 302, 301, 300, 
           299, 298, 297, 295, 292, 291, 290, 288, 287, 283, 279, 277, 276, 275, 274, 273, 271, 269, 268, 267, 265, 263, 262, 260, 259, 258, 256, 252, 251, 249, 246, 244, 243, 242, 241, 238, 
           236, 235, 233, 232, 231, 230, 229, 224, 221, 219, 218, 217, 216, 215, 213, 210, 208, 207, 203, 201, 199, 198, 192, 180, 175, 172, 167, 165, 158, 154, 150, 147, 145, 133, 127, 
           117, 113, 112, 108, 107, 103, 88, 79, 73, 70, 69, 65, 51, 46, 37, 25, 22, 21, 19, 18, 15, 12, 6, 3]
end_of_bad = dff.shape[0]

if len(bad_roi) == 0:
    for i in range(dff.shape[0]): 
        if not i in good_roi:
            bad_roi.append(i)
elif len(good_roi) == 0:
    for i in range(dff.shape[0]):
        if not i in bad_roi:
            good_roi.append(i)
else:
    for i in range(end_of_bad):
        if not i in bad_roi:
            good_roi.append(i)
    for i in range(dff.shape[0]): 
        if not i in good_roi and i not in bad_roi:
            bad_roi.append(i)
    

save_lables(ops, good_roi, bad_roi)


run_manual_postprocess.run(session_data_path)

if need_plot:
    dff = read_dff(ops, 'manual_qc_results')
    fs = 30
    time_min = np.arange(0 , dff.shape[1])/(fs*60)
    time_s = np.arange(0 , dff.shape[1])/(fs)
    
    total_neurons = dff.shape[0]
    start = 0
    for itr in range(1, total_neurons//150+2):
        end = min(150*itr, total_neurons)
        curr_max = 0 
        shift = np.nanmin(dff[0, :])
    
        for i in range(start, end):
            curr_dff = dff[i , :]
            median = np.nanmedian(curr_dff)
            curr_dff = curr_dff - median
            curr_min = np.nanmin(curr_dff)
            shift = shift + np.abs(curr_min) + curr_max
            curr_max = np.nanmax(curr_dff)
            if i == start:
                fig = px.line(x = list(time_min), y = list(curr_dff+shift) , labels = 'Neuron id= ' + str(i))
            else:
                fig.add_trace(go.Scatter(x = list(time_min), y = list(curr_dff+shift), mode='lines', name= 'Neuron id= ' + str(i)))
                
        fig.update_layout(title='Raw dff traces', xaxis_title='Time (min)', yaxis_title='dff traces')
              
        fig.write_html(output_dir_onedrive  + data_date + '_raw_traces_good_rois_part' + str(itr) + '.html')
        start = end
            
dff = read_dff(ops, 'manual_qc_results')
print('Number of good ROIs: ', dff.shape[0])

