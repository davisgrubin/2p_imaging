# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:13:56 2025

@author: saminnaji3
"""

import ReadTime

# file_path = 'C:/Users/saminnaji3/Downloads/SA12_20250922_Random_Vis_Only_cam0_run000_20250922_080331.camlog'
# vol_path = 'C:/Users/saminnaji3/Downloads/passive/SA12_LG/SA12_20250922'

# vol_path is the path of folder where voltage recording is saved inside
# put the camlog file in the same folder and name it as camlog_file and then file_path will be the same folder path
vol_path = 'C:/Users/saminnaji3/Downloads/Check_Align_session'
file_path = 'C:/Users/saminnaji3/Downloads/Check_Align_session'

# time_flir will be the time of each frame of video recording
# time_neuro will be the time of each frame of 2p recording
time_flir, time_neuro = ReadTime.run(vol_path, file_path)


