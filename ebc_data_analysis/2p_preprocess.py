import os
from utils.functions import processing_files
from configurations import conf

mice = [folder for folder in os.listdir('./data/imaging/')]
for mouse_name in mice:
    if mouse_name[0]=='.':
        continue 
    session_folder = [folder for folder in os.listdir(f'./data/imaging/{mouse_name}/')]
    print(session_folder)
    print(mice)
    for session_date in session_folder:
        if session_date[0]=='.':
            continue
        # try:
        processing_files(bpod_file = f"./data/imaging/{mouse_name}/{session_date}/bpod_session_data.mat", 
                        raw_voltage_file = f"./data/imaging/{mouse_name}/{session_date}/raw_voltages.h5", 
                        dff_file = f"./data/imaging/{mouse_name}/{session_date}/dff.h5", 
                        save_path = f"./data/imaging/{mouse_name}/{session_date}/saved_trials.h5", exclude_start=10, exclude_end=10)
        # except:
        #     print(f"the session{session_date} has some data issues")
