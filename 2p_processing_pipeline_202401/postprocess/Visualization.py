#!/usr/bin/env python3

from plot.fig1_mask import plot_fig1
from plot.fig2_raw_traces import plot_fig2
#from plot.fig3_spike_trigger_average import plot_fig3
from plot.fig4_align_grating import plot_fig4
from plot.fig5_align_error import plot_fig5

#from plot.san1_motion_offsets import plot_san1


# main function for trialization

def run(ops):
    print('===============================================')
    print('================ visualization ================')
    print('===============================================')
    
    plot_fig1(ops)
    plot_fig2(ops)
    #plot_fig3(ops)
    plot_fig4(ops)
    plot_fig5(ops)
