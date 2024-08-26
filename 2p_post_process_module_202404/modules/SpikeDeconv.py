import matplotlib.pyplot as plt
from suite2p.extraction import oasis


# def read_ops(session_data_path):
#     ops = np.load(
#         os.path.join(session_data_path, 'suite2p', 'plane0', 'ops.npy'),
#         allow_pickle=True).item()
#     ops['save_path0'] = os.path.join(session_data_path)
#     return ops


def read_raw_voltages(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'raw_voltages.h5'),
        'r')
    try:
        vol_time = np.array(f['raw']['vol_time'])
        vol_start = np.array(f['raw']['vol_start'])
        vol_stim_vis = np.array(f['raw']['vol_stim_vis'])
        vol_hifi = np.array(f['raw']['vol_hifi'])
        vol_img = np.array(f['raw']['vol_img'])
        vol_stim_aud = np.array(f['raw']['vol_stim_aud'])
        vol_flir = np.array(f['raw']['vol_flir'])
        vol_pmt = np.array(f['raw']['vol_pmt'])
        vol_led = np.array(f['raw']['vol_led'])

    except:
        vol_time = np.array(f['raw']['vol_time'])
        vol_start = np.array(f['raw']['vol_start_bin'])
        vol_stim_vis = np.array(f['raw']['vol_stim_bin'])
        vol_img = np.array(f['raw']['vol_img_bin'])
        vol_hifi = np.zeros_like(vol_time)
        vol_stim_aud = np.zeros_like(vol_time)
        vol_flir = np.zeros_like(vol_time)
        vol_pmt = np.zeros_like(vol_time)
        vol_led = np.zeros_like(vol_time)

    f.close()
    return [vol_time, vol_start, vol_stim_vis, vol_img,
            vol_hifi, vol_stim_aud, vol_flir,
            vol_pmt, vol_led]


def read_dff(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'dff.h5'), 'r')
    dff = np.array(f['dff'])
    f.close()
    return dff


# i = 5
# plt.plot(vol_img, dff[i,:])

def plot(i):
    plt.plot(vol_img, dff[i, :])
    plt.savefig('example')


def spike_detect(
        ops,
        dff
):
    # oasis for spike detection.
    spikes = oasis(
        F=dff,
        batch_size=ops['batch_size'],
        tau=ops['tau'],
        fs=ops['fs'])
    return spikes


def run(ops, dff):
    print('===================================================')
    print('=============== Deconvolving Spikes ===============')
    print('===================================================')

    stats = read_raw_voltages(ops)
    dff = read_dff(ops)
    plot(5)

    return spike_detect(ops, dff)
