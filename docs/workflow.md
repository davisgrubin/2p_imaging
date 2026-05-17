# 2p Data Processing Overview

The starting point for all our 2p imaging based analyses are anatomical and functional images from which ROIs (regions of interest) are extracted. After applying QC and labeling we create dF/F time series to measure activity of these ROIs, which are aligned with experimental recordings of stimuli presentation, behavior such as licking, or other events to perform our downstream analyses.

<div style="max-width: 860px; margin: 1.25rem auto 1rem;">
  <div style="display: grid; grid-template-columns: 1fr 90px 1fr; align-items: start; column-gap: 1rem;">
    <div>
      <div style="display: flex; gap: 0.75rem; justify-content: flex-start; margin-bottom: 0.5rem; font-weight: 700;">
        <div style="width: 48%; text-align: center; color: #8b1e3f;">Anatomical (Red)</div>
        <div style="width: 48%; text-align: center; color: #1f6f43;">Functional (Green)</div>
      </div>
      <div style="display: flex; gap: 0.75rem; align-items: flex-start;">
        <div style="width: 48%; background: #fff; border: 1px solid #d8dee9; box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12); border-radius: 8px; overflow: hidden;">
          <img src="../assets/mc11_mean_anatomical.png" alt="Mean anatomical channel image" style="display: block; width: 100%; height: auto;" />
        </div>
        <div style="width: 48%; background: #fff; border: 1px solid #d8dee9; box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12); border-radius: 8px; overflow: hidden;">
          <img src="../assets/mc11_mean_functional.png" alt="Mean functional channel image" style="display: block; width: 100%; height: auto;" />
        </div>
      </div>
    </div>
    <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
      <div style="font-size: 3rem; line-height: 1; color: #64748b;">&#8594;</div>
    </div>
    <div>
      <div style="margin-bottom: 0.5rem; text-align: center; font-weight: 700; color: #1f2937;">Example dF/F time series</div>
      <div style="background: #fff; border: 1px solid #d8dee9; box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12); border-radius: 8px; overflow: hidden; padding: 0.75rem 0.9rem;">
        <svg viewBox="0 0 360 240" width="100%" height="auto" aria-label="Example stacked dF/F traces">
          <rect x="0" y="0" width="360" height="240" fill="white"></rect>
          <line x1="0" y1="50" x2="360" y2="50" stroke="#e5e7eb" stroke-width="1"></line>
          <line x1="0" y1="120" x2="360" y2="120" stroke="#e5e7eb" stroke-width="1"></line>
          <line x1="0" y1="190" x2="360" y2="190" stroke="#e5e7eb" stroke-width="1"></line>
          <path d="M0 48 L20 47 L40 45 L60 50 L80 44 L100 42 L120 49 L140 46 L160 30 L180 20 L200 33 L220 47 L240 44 L260 41 L280 46 L300 49 L320 43 L340 45 L360 47" fill="none" stroke="#1f6f43" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>
          <path d="M0 118 L20 116 L40 120 L60 114 L80 111 L100 119 L120 116 L140 112 L160 108 L180 95 L200 82 L220 98 L240 115 L260 118 L280 112 L300 110 L320 116 L340 113 L360 118" fill="none" stroke="#2f855a" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>
          <path d="M0 188 L20 186 L40 190 L60 187 L80 180 L100 171 L120 184 L140 188 L160 181 L180 176 L200 172 L220 165 L240 176 L260 187 L280 184 L300 179 L320 182 L340 188 L360 186" fill="none" stroke="#38a169" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>
          <text x="6" y="18" font-size="13" fill="#475569">ROI 1</text>
          <text x="6" y="88" font-size="13" fill="#475569">ROI 2</text>
          <text x="6" y="158" font-size="13" fill="#475569">ROI 3</text>
          <text x="300" y="232" font-size="12" fill="#64748b">time</text>
        </svg>
      </div>
    </div>
  </div>
</div>

## 5 Step Data Processing Workflow

<div class="overview-flowchart">
  <img src="../assets/flowchart-large.svg" alt="2p imaging workflow flowchart" />
</div>

<p class="workflow-heading"><strong>Processing</strong></p>

<ol class="workflow-steps">
  <li>Prep</li>
  <li>Suite2p</li>
</ol>

<p class="workflow-heading"><strong>Postprocessing</strong></p>

<ol start="3" class="workflow-steps">
  <li>QC</li>
  <li>Label</li>
  <li>dF/F</li>
</ol>

The following sections provide more detail on each data processing step, files generated, and definitions of the fields and data contained within.

The two main locations where data is stored are PACE project storage, where files are uploaded from experimental or lab PCs after recording sessions, and the long-term CEDAR data storage.

For more information on the technical specifications of the 2p imaging rig and experimental recording setup, see the respective OneDrive documentation links when they are added here in the future.
