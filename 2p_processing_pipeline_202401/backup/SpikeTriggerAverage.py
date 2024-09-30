import numpy as np

# compute spike triggered average.
def get_sta(
        dff,
        img_time,
        spikes,
        l_frames,
        r_frames,
        percentile = 0.01,
        ):
    all_tri_dff = []
    all_tri_time = []
    # check neuron one by one.
    for neu_idx in range(dff.shape[0]):
        neu_tri_dff = []
        neu_tri_time = []
        # find the largest num_spikes spikes.
        neu_spike_time = np.nonzero(spikes[neu_idx,:])[0]
        neu_spike_amp = spikes[neu_idx,neu_spike_time]
        if len(neu_spike_amp) > 0:
            num_spikes = int(len(neu_spike_amp)*percentile)
            neu_spike_time = neu_spike_time[np.argsort(-neu_spike_amp)[:num_spikes]]
        # find spikes.
        if len(neu_spike_time) > 0:
            for neu_t in neu_spike_time:
                if (neu_t - l_frames > 0 and
                    neu_t + r_frames < dff.shape[1]):
                    f = dff[neu_idx,neu_t - l_frames:neu_t + r_frames]
                    f = f.reshape(1, -1)
                    f = f / (spikes[neu_idx,neu_t]+1e-5)
                    neu_tri_dff.append(f)
                    t = img_time[neu_t - l_frames:neu_t + r_frames] - img_time[neu_t]
                    neu_tri_time.append(t)
            if len(neu_tri_dff) > 0:
                neu_tri_dff = np.concatenate(neu_tri_dff, axis=0)
                neu_tri_time = np.mean(neu_tri_time, axis=0)
            else:
                neu_tri_dff = []
                neu_tri_time = []
        else:
            neu_tri_dff = []
            neu_tri_time = []
        all_tri_dff.append(neu_tri_dff)
        all_tri_time.append(neu_tri_time)
    return all_tri_dff, all_tri_time

# spike triggered average.
self.all_tri_dff, self.all_tri_time = get_sta(
        dff, self.img_time, self.spikes, self.l_frames, self.r_frames)
def spike_tri_average(self, ax, roi_id):
    if isinstance(self.all_tri_dff[roi_id], np.ndarray):
        dff_mean = np.mean(self.all_tri_dff[roi_id], axis=0)
        dff_sem = sem(self.all_tri_dff[roi_id], axis=0)
        ax.plot(
            self.all_tri_time[roi_id],
            dff_mean,
            color='grey')
        ax.fill_between(
            self.all_tri_time[roi_id],
            dff_mean - dff_sem,
            dff_mean + dff_sem,
            color='grey', alpha=0.2)
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(np.min(self.all_tri_time[roi_id]),
                    np.max(self.all_tri_time[roi_id]))
        ax.set_xlabel('time since center spike (ms)')
        ax.set_ylabel('z-scored df/f (mean$\pm$sem)')
        ax.set_title('ROI # {} spike trigger average'.format(
            str(roi_id).zfill(4)))