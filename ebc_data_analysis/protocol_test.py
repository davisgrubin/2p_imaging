from utils.functions import read_bpod_mat_data, trial_label_fec
# just run a single processing on the curropted .mat file


bpod_file = './data/beh/E4L7/raw/E5LG_EBC_V_3_12_20241024_160443mat'

bpod_data_processed = read_bpod_mat_data(bpod_file=bpod_file)
trials = trial_label_fec(bpod_data_processed)

# print(bpod_data_processed['trial_LED_ON'])
# print(len(bpod_data_processed['trial_LED_ON']))
print(trials['59'])
