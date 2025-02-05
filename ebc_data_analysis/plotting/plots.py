import numpy as np
import matplotlib.pyplot as plt
import h5py as hpy

def plotting_roi_masks(masks_func):
    masks = {}

    for x in range(len(masks_func[0])):
        for y in range(len(masks_func)):
            if masks_func[y][x] > 0.2:
                pixel_value = masks_func[y][x]
                int_pixel_value = int(pixel_value)

                # Initialize the mask for this pixel value if it doesn't exist
                if int_pixel_value not in masks:
                    masks[int_pixel_value] = np.zeros((len(masks_func), len(masks_func[0])))

                # Set the value at the corresponding position to 1.0
                masks[int_pixel_value][y][x] = 1.0
    return masks

def plot_masks_functions(mask_file, ax_max, ax_mean, ax_mask):

    with hpy.File(mask_file) as f:
        max_func = f["max_func"][:]
        mean_func = f["mean_func"][:]
        masks_func = f["masks_func"][:]  # Assuming it contains multiple masks.
        f.close()

    # Plot max function
    ax_max.matshow(max_func, cmap='magma')
    ax_max.axis('off')  # Turn off axes for cleaner visuals

    # Plot mean function
    ax_mean.matshow(mean_func, cmap='viridis')
    ax_mean.axis('off')

    # Plot first mask
    ax_mask.matshow(masks_func, cmap='plasma', alpha = 0.7)  # Display the first mask
    ax_mask.axis('off')

def plot_trial_averages_sig(trials, aligned_time, crp_avg, crp_sem, crn_avg, crn_sem, title_suffix, event, pooled, ax):
    x_min, x_max = -150, 550

    if title_suffix == "Short":
        suffix_color = "blue"
    if title_suffix == "Long":
        suffix_color = "lime"

    # Extract time and trial id
    time = aligned_time[list(aligned_time.keys())[0]]
    id = list(aligned_time.keys())[0]

    # Mask and filter data
    mask = (time >= x_min) & (time <= x_max)
    crp_filtered = crp_avg[mask]
    crp_sem_filtered = crp_sem[mask]
    crn_filtered = crn_avg[mask]
    crn_sem_filtered = crn_sem[mask]

    # Calculate y-axis limits
    y_min = min(
        np.min(crp_filtered - crp_sem_filtered),
        np.min(crn_filtered - crn_sem_filtered)
    )
    y_max = max(
        np.max(crp_filtered + crp_sem_filtered),
        np.max(crn_filtered + crn_sem_filtered)
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    trial_type = "Pooled" if pooled else "Grand"
    ax.set_title(f"{trial_type} Average of {title_suffix} Trials for {event} significant ROIs")
    ax.set_xlabel("Time for LED Onset (ms)")
    ax.set_ylabel("Mean df/f (+/- SEM)")

    ax.plot(time, crp_avg, label="CR+", color="red")
    ax.fill_between(time, crp_avg - crp_sem, crp_avg + crp_sem, color="red", alpha=0.1)

    ax.plot(time, crn_avg, label="CR-", color="blue")
    ax.fill_between(time, crn_avg - crn_sem, crn_avg + crn_sem, color="blue", alpha=0.1)

    # Add shaded regions for LED and AirPuff
    ax.axvspan(0, trials[id]["LED"][1] - trials[id]["LED"][0], color="gray", alpha=0.5, label="LED")
    ax.axvspan(trials[id]["AirPuff"][0] - trials[id]["LED"][0], trials[id]["AirPuff"][1] - trials[id]["LED"][0], 
               color=suffix_color, alpha=0.5, label="AirPuff")

    ax.legend()


def plot_trial_averages_side_by_side(
    ax1, ax2, n_value_p1, n_value_n1, aligned_time1, crp_avg1, crp_sem1, crn_avg1, crn_sem1, 
    n_value_p2, n_value_n2, aligned_time2, crp_avg2, crp_sem2, crn_avg2, crn_sem2, 
    trials, title_suffix1, title_suffix2, pooled=False
):
    x_min, x_max = -100, 600
    colors = {"Short": "blue", "Long": "lime"}
    
    # Calculate global y-axis limits
    global_y_min = float("inf")
    global_y_max = float("-inf")

    # First loop to determine global y-axis limits
    for n_value_p, n_value_n, aligned_time, crp_avg, crp_sem, crn_avg, crn_sem in [
        (n_value_p1, n_value_n1, aligned_time1, crp_avg1, crp_sem1, crn_avg1, crn_sem1),
        (n_value_p2, n_value_n2, aligned_time2, crp_avg2, crp_sem2, crn_avg2, crn_sem2),
    ]:
        time = aligned_time[list(aligned_time.keys())[0]]
        mask = (time >= x_min) & (time <= x_max)
        crp_filtered = crp_avg[mask]
        crp_sem_filtered = crp_sem[mask]
        crn_filtered = crn_avg[mask]
        crn_sem_filtered = crn_sem[mask]

        global_y_min = min(global_y_min, np.min(crp_filtered - crp_sem_filtered), np.min(crn_filtered - crn_sem_filtered))
        global_y_max = max(global_y_max, np.max(crp_filtered + crp_sem_filtered), np.max(crn_filtered + crn_sem_filtered))

    # Second loop to plot data
    # Second loop to plot data
    for (ax, n_value_p, n_value_n, aligned_time, crp_avg, crp_sem, crn_avg, crn_sem, title_suffix) in [
        (ax1, n_value_p1, n_value_n1, aligned_time1, crp_avg1, crp_sem1, crn_avg1, crn_sem1, title_suffix1),
        (ax2, n_value_p2, n_value_n2, aligned_time2, crp_avg2, crp_sem2, crn_avg2, crn_sem2, title_suffix2),
    ]:

        suffix_color = colors.get(title_suffix, "gray")

        # Extract time and trial id
        time = aligned_time[list(aligned_time.keys())[0]]
        id = list(aligned_time.keys())[0]

        # Mask and filter data
        mask = (time >= x_min) & (time <= x_max)
        crp_filtered = crp_avg[mask]
        crn_filtered = crn_avg[mask]

        # Plot data
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(global_y_min, global_y_max)  # Use global y-axis limits
        trial_type = "Pooled" if pooled else "Grand"
        n_name = "df/f signals" if pooled else "Trial averages"
        ax.set_title(f"{trial_type} Average of df/f signals for {title_suffix} Trials")
        ax.set_xlabel("Time for LED Onset (ms)")
        ax.set_ylabel("Mean df/f (+/- SEM)")

        ax.plot(time, crp_avg, label=f"CR+- : {n_value_p} {n_name}", color="red")
        ax.fill_between(time, crp_avg - crp_sem, crp_avg + crp_sem, color="red", alpha=0.1)

        ax.plot(time, crn_avg, label=f"CR- : {n_value_n} {n_name}", color="blue")
        ax.fill_between(time, crn_avg - crn_sem, crn_avg + crn_sem, color="blue", alpha=0.1)

        # Add shaded regions for LED and AirPuff
        ax.axvspan(0, trials[id]["LED"][1] - trials[id]["LED"][0], color="gray", alpha=0.5, label="LED")
        ax.axvspan(trials[id]["AirPuff"][0] - trials[id]["LED"][0], trials[id]["AirPuff"][1] - trials[id]["LED"][0], 
                   color=suffix_color, alpha=0.5, label="AirPuff")

        ax.legend()


def plot_heatmaps_side_by_side(heat_arrays, aligned_times, titles, trials, color_maps, axes):
    for i, (arr, aligned_time, title, color_map, ax) in enumerate(zip(heat_arrays, aligned_times, titles, color_maps, axes)):
        time_array = aligned_time[list(aligned_time.keys())[0]]
        print(time_array)
        id = list(aligned_time.keys())[0]
        time_limits = [-100, 600]

        # Set the min and max for the color scale based on the actual data
        vmin = arr.min()
        vmax = arr.max()

        # Plot heatmap with dynamic color scaling
        im = ax.imshow(arr, aspect="auto", cmap=color_map, origin="upper",
                       extent=[time_limits[0], time_limits[-1], 0, arr.shape[0]], vmin=vmin, vmax=vmax)

        # Add colorbar
        plt.colorbar(im, ax=ax)

        airpuff_color = "lime" if "Long" in title else "blue"

        ax.axvline(trials[id]["AirPuff"][0] - trials[id]["LED"][0], color=airpuff_color, linestyle=':', linewidth=2, label="AirPuff")  # Dotted line for AirPuff
        ax.axvline(trials[id]["AirPuff"][1] - trials[id]["LED"][0], color=airpuff_color, linestyle=':', linewidth=2)  # Dotted line for AirPuff end
        ax.axvline(0, color="gray", linestyle=':', linewidth=2, label="LED")  # Dotted line for LED start
        ax.axvline(trials[id]["LED"][1] - trials[id]["LED"][0], color="gray", linestyle=':', linewidth=2)  # Dotted line for LED end

        ax.set_xlim(-100, 600)
        ax.set_title(title)
        ax.set_xlabel("Time for LED Onset(ms)")
        ax.set_ylabel("Number of ROIs")

def plot_single_condition_scatter(baseline_avg, event_avg, significant_rois, color, label, title):

    plt.figure(figsize=(8, 6))

    base_plot  = []
    event_plot = []

    for roi in baseline_avg:
        roi_color = 'lime' if roi in significant_rois else color
        base = baseline_avg[roi]
        event = event_avg[roi]
        # base_plot .append(np.nanmean(base, axis=0))
        # event_plot.append( np.nanmean(event, axis=0))
        base_plot = np.nanmean(base, axis=0)
        event_plot = np.nanmean(event, axis=0)

        plt.scatter(base_plot, event_plot, color=roi_color)
    # Labels, title, and legend
    plt.xlabel("Baseline Interval Average dF/F")
    plt.ylabel("Event Interval Average dF/F")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()

def plot_fec_trial(ax, time, mean1, std1, mean0, std0, led, airpuff, y_lim, title):
    ax.plot(time, mean1, label="CR+", color="red")
    ax.fill_between(time, mean1 - std1, mean1 + std1, color="red", alpha=0.1)
    ax.plot(time, mean0, label="CR-", color="blue")
    ax.fill_between(time, mean0 - std0, mean0 + std0, color="blue", alpha=0.1)
    ax.axvspan(0, led[1] - led[0], color="gray", alpha=0.5, label="LED")
    ax.axvspan(airpuff[0] - led[0], airpuff[1] - led[0], color="blue" if "Short" in title else "lime", alpha=0.5, label="AirPuff")
    ax.set_title(title)
    ax.set_xlabel("Time for LED Onset (ms)")
    ax.set_ylabel("FEC")
    ax.set_xlim(-100, 600)
    ax.set_ylim(y_lim, 1.1)
    ax.legend()


def plot_histogram(ax, data, bins, color, edgecolor, alpha, title, xlabel, ylabel):
    ax.hist(data, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_scatter(ax, x_data, y_data, color, alpha, label, title, xlabel, ylabel):
    ax.scatter(x_data, y_data, color=color, alpha=alpha, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_hexbin(ax, x_data, y_data, gridsize, cmap, mincnt, alpha, colorbar_label, title, xlabel, ylabel):
    hb = ax.hexbin(x_data, y_data, gridsize=gridsize, cmap=cmap, mincnt=mincnt, alpha=alpha)
    # cbar = plt.colorbar(hb, ax=ax)
    # cbar.set_label(colorbar_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
