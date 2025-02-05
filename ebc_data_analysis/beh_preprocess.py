from utils.functions import processing_beh
#from configurations import conf
import os

beh_folder = "./data/beh"
mice = [folder for folder in os.listdir(beh_folder)] 
for mouse in mice:
    all_sessions = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(beh_folder, mouse,"raw"))]
    # print(all_sessions)
    i = 0
    for session_date in all_sessions:
        try:
            processed_folder = os.path.join(beh_folder, mouse, "processed")
            if not os.path.exists(processed_folder):
                os.mkdir(processed_folder) 

            processing_beh(bpod_file = f"./data/beh/{mouse}/raw/{session_date}.mat", 
            save_path = f"./data/beh/{mouse}/processed/{session_date}.h5", 
            exclude_start=0, exclude_end=0)
        except:
            # print("idk why")
            i += 1
    print("success rate for behavior file processing:", 1 - i/len(all_sessions))
    print(f"{mouse} completed")
