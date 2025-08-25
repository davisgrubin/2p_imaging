import os
from utils.functions import processing_files

mice = [folder for folder in os.listdir('./data/imaging/')]
for mouse_name in mice:
    print(f'processing {mouse_name}')
    if mouse_name[0]=='.':
        continue 
    session_folder = [folder for folder in os.listdir(f'./data/imaging/{mouse_name}/')]
    for session_date in session_folder:
        print(f'processing session {session_date}')
        if session_date[0]=='.':
            continue
        # try:
        processing_files(bpod_file = f"./data/imaging/{mouse_name}/{session_date}/bpod_session_data.mat", 
                    raw_voltage_file = f"./data/imaging/{mouse_name}/{session_date}/raw_voltages.h5", 
                    dff_file = f"./data/imaging/{mouse_name}/{session_date}/dff.h5", 
                    save_path = f"./data/imaging/{mouse_name}/{session_date}/saved_trials.h5", exclude_start=10, exclude_end=10)

        print(f"PDF successfully saved: {session_date}")
        # except Exception as e:
            # print(f"the session{session_date} has some data issues {e}")
