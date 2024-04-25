# tentative function to hard code trial types for passive sessions.

def add_jitter_flag(ops, neural_trials):
    from sklearn.mixture import GaussianMixture
    def frame_dur(stim, time):
        diff_stim = np.diff(stim, prepend=0)
        idx_up   = np.where(diff_stim == 1)[0]
        idx_down = np.where(diff_stim == -1)[0]
        dur_high = time[idx_down] - time[idx_up]
        dur_low  = time[idx_up[1:]] - time[idx_down[:-1]]
        return [dur_high, dur_low]
    def get_mean_std(isi):
        gmm = GaussianMixture(n_components=2)
        gmm.fit(isi.reshape(-1,1))
        std = np.mean(np.sqrt(gmm.covariances_.flatten()))
        return std
    thres = 25
    jitter_flag = []
    for i in range(len(neural_trials)):
        stim = neural_trials[str(i)]['stim']
        time = neural_trials[str(i)]['time']
        [_, isi] = frame_dur(stim, time)
        std = get_mean_std(isi)
        jitter_flag.append(std)
    jitter_flag = np.array(jitter_flag)
    jitter_flag[jitter_flag<thres] = 0
    jitter_flag[jitter_flag>thres] = 1
    np.save(os.path.join(ops['save_path0'], 'jitter_flag.npy'), jitter_flag)