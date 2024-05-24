def spont_evoke(self, ax, roi_id):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    roi_id = 66
    
    x = np.linspace(0,
        np.max([np.max(self.spike_spont[roi_id,:]),
                np.max(self.spike_evoke[roi_id,:])]), 200)
    alpha_spont, loc_spont, beta_spont = gamma.fit(self.spike_spont[roi_id,:])
    alpha_evoke, loc_evoke, beta_evoke = gamma.fit(self.spike_evoke[roi_id,:])
    pdf_spont = gamma.pdf(x, alpha_spont, loc=loc_spont, scale=beta_spont)
    pdf_evoke = gamma.pdf(x, alpha_evoke, loc=loc_evoke, scale=beta_evoke)
    _, color1, color2 = get_roi_label_color(self.labels, roi_id)
    ax.plot(x, pdf_spont, color=color1, label='spontaneous')
    ax.plot(x, pdf_evoke, color=color2, label='evoked')
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0, np.max(x))
    ax.set_xlabel('df/f (z-scored)')
    ax.set_ylabel('probability density')
    ax.legend(loc='upper right')