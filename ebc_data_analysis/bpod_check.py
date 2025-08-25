import scipy.io as sio
import numpy as np

def check_keys(d):
    for key in d:
        if isinstance(d[key], sio.matlab.mat_struct):
            d[key] = todict(d[key])
    return d

def todict(matobj):
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mat_struct):
            d[strg] = todict(elem)
        elif isinstance(elem, np.ndarray):
            d[strg] = tolist(elem)
        else:
            d[strg] = elem
    return d

def tolist(ndarray):
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, sio.matlab.mat_struct):
            elem_list.append(todict(sub_elem))
        elif isinstance(sub_elem, np.ndarray):
            elem_list.append(tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list

# bpod_file = "./data/beh/1/raw/2023-01-17.mat"
bpod_file = './data/imaging/E4L7-SD/20241105_crbl_ebc_SleepDep/bpod_session_data.mat'
raw = sio.loadmat(bpod_file, struct_as_record=False, squeeze_me=True)
raw = check_keys(raw)
raw_session_data = check_keys(raw)['SessionData']

# print(raw['SleepDeprived'])

for item in raw_session_data:
    print(item)

print(raw_session_data['SleepDeprived'])

