from utils.functions import processing_beh
#from configurations import conf
import os
import numpy as np

beh_folder = "./data/beh"
raw_folder = "raw"
mice = [folder for folder in os.listdir(beh_folder) if folder != ".DS_Store"] 
for mouse in mice:
    # if mouse in ['E4L7', 'E5LG', 'E6LG', 'E1VT', 'E3VT']:
    #     continue

    if mouse != 'E5LG':
        continue

    all_sessions = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(beh_folder, mouse,raw_folder)) if not file.startswith(".")]
    all_sessions.sort()
    for session_date in all_sessions:
        # if 'V_3_18' not in session_date:
        #     continue
        print(session_date)
        processed_folder = os.path.join(beh_folder, mouse, "processed")
        if not os.path.exists(processed_folder):
            os.mkdir(processed_folder) 

        processing_beh(bpod_file = f"./data/beh/{mouse}/{raw_folder}/{session_date}.mat", 
        save_path = f"./data/beh/{mouse}/processed/{session_date}.h5", 
        exclude_start=0, exclude_end=0)
        print(f"{session_date} processed")

    print(f"{mouse} completed")
