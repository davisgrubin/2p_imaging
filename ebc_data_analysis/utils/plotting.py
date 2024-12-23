import numpy as np
import matplotlib.pyplot as plt

from functions import roi_group_analysis

def plotting_trials_cut(trials , trials_under_test, ROI_under_test, buffer=500):
    for trial_id in trials_under_test:
        plt.figure(figsize=(20, 6))

        # Determine the time range with buffer
        start_time = min(trials[trial_id]["LED"][0], trials[trial_id]["AirPuff"][0]) - buffer
        end_time = max(trials[trial_id]["LED"][1], trials[trial_id]["AirPuff"][1]) + buffer

        # Plot each ROI with its corresponding color
        for roi in ROI_under_test:
            plt.plot(trials[trial_id]["time"][:], trials[trial_id]["dff"][roi], label=f"ROI {roi}")

        # Add vertical lines for events
        plt.axvline(x=trials[trial_id]["LED"][0], color='purple', linestyle='--', label="LED Start")
        plt.axvline(x=trials[trial_id]["LED"][1], color='purple', linestyle='--', label="LED End")
        plt.axvline(x=trials[trial_id]["AirPuff"][0], color='gray', linestyle='--', label="Air Puff Start")
        plt.axvline(x=trials[trial_id]["AirPuff"][1], color='gray', linestyle='--', label="Air Puff End")
        plt.axvline(x=trials[trial_id]["ITI"][0], color='green', linestyle='--', label="ITI Start")
        plt.axvline(x=trials[trial_id]["ITI"][1], color='green', linestyle='--', label="ITI End")

        # Set x-axis limits
        plt.xlim(start_time, end_time)

        # Add legend
        plt.legend()

        # Add titles and labels
        plt.title(f"Trial ID: {trial_id}")
        plt.xlabel("Time (s)")
        plt.ylabel("dF/F")

        # Show the plot
        plt.show()

def plotting_trials(trials, trials_under_test , ROI_under_test):

    for id in trials_under_test:
        plt.figure(figsize=(20, 6))

        # Plot each ROI with its corresponding color
        for roi in ROI_under_test:
            plt.plot(trials[id]["time"][:] , trials[id]["dff"][roi], label=f"ROI {roi}")

        # Add legend

        plt.axvline(x=trials[id]["LED"][0], color='purple', linestyle='--', label="LED")
        plt.axvline(x=trials[id]["LED"][1], color='purple', linestyle='--')
        plt.axvline(x=trials[id]["AirPuff"][0], color='gray', linestyle='--', label="AirPuff")
        plt.axvline(x=trials[id]["AirPuff"][1], color='gray', linestyle='--')
        plt.axvline(x=trials[id]["ITI"][0], color='green', linestyle='--', label="ITI")
        plt.axvline(x=trials[id]["ITI"][1], color='green', linestyle='--')


        plt.legend()

        plt.title(f"Trial ID: {id}")
        plt.xlabel("Time")
        plt.ylabel("dF/F")
        plt.show()

def plotting_avg_cut(trials , trials_under_test, buffer , ROI_under_test):
    for trial_id in trials_under_test:

        avg_dff , std_dff = roi_group_analysis(trials , trial_id=trial_id , roi_group=ROI_under_test)

        plt.figure(figsize=(20, 6))

        # Determine the time range with buffer
        start_time = trials[trial_id]["LED"][0] - buffer
        end_time = max(trials[trial_id]["LED"][1], trials[trial_id]["AirPuff"][1]) + buffer

        plt.plot(trials[trial_id]["time"][:] , avg_dff , label = "average")
        plt.plot(trials[trial_id]["time"][:] , std_dff , label = "std")

        # Add vertical lines for events
        plt.axvline(x=trials[trial_id]["LED"][0], color='purple', linestyle='--', label="LED")
        plt.axvline(x=trials[trial_id]["LED"][1], color='purple', linestyle='--')
        plt.axvline(x=trials[trial_id]["AirPuff"][0], color='gray', linestyle='--', label="Air Puff")
        plt.axvline(x=trials[trial_id]["AirPuff"][1], color='gray', linestyle='--')
        plt.axvline(x=trials[trial_id]["ITI"][0], color='green', linestyle='--', label="ITI")
        plt.axvline(x=trials[trial_id]["ITI"][1], color='green', linestyle='--')

        # Set x-axis limits
        plt.xlim(start_time, end_time)

        # Add legend
        plt.legend()

        # Add titles and labels
        plt.title(f"Trial ID: {trial_id}")
        plt.xlabel("Time (s)")
        plt.ylabel("dF/F")

        # Show the plot
        plt.show()