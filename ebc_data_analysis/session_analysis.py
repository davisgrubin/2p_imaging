import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
from matplotlib.gridspec import GridSpec
import h5py

from utils.alignment import sort_numbers_as_strings, fec_zero
from utils.indication import CR_stat_indication

def get_isi_value(trial):
    """Get the actual ISI value and return the rounded/grouped value"""
    airpuff = trial["AirPuff"][0] - trial["LED"][0]
    return airpuff

def group_isi_value(isi_value):
    """Group ISI values into expected categories (200, 300, 400) with ±10 tolerance"""
    if abs(isi_value - 200) <= 10:
        return 200
    elif abs(isi_value - 300) <= 10:
        return 300
    elif abs(isi_value - 400) <= 10:
        return 400
    else:
        # Round to nearest 10 for unexpected values
        return round(isi_value / 10) * 10

beh_folder = "./data/beh"
early_trials_check = "early"
static_threshold = 0.03

mice = [folder for folder in os.listdir(beh_folder) if folder != ".DS_Store"] 
# mice = ['E6LG']
# Sort the mice according to alphabetical order
mice.sort()

mice_cr_performance_file = './outputs/beh/mice_cr_performance_and_isi.pdf'
x_grid = 4
y_grid = len(mice) * 4  # Four rows per mouse (CR, ISI, Test Type, Trial Count)
fig = plt.figure(figsize=(x_grid * 5, y_grid * 3))  # Adjusted height per row
gs = GridSpec(y_grid, x_grid)

for mouse_i, mouse in enumerate(mice):
    # Set Long_airpuff parameters for specific mice

    folder_path = f'./outputs/beh/evaluation/{mouse}'
    os.makedirs(folder_path, exist_ok=True)

    session_path = os.path.join(beh_folder, mouse, "processed")

    all_sessions = sorted([
        os.path.splitext(file)[0]
        for file in os.listdir(session_path)
        if file.endswith('.h5')  # Only include .h5 files but don't skip hidden files
    ])

    # all_sessions = all_sessions[-20:]

    if not os.path.exists(f'./outputs/beh/{mouse}'):
        os.mkdir(f'./outputs/beh/{mouse}')
        print(f"data folder '{mouse}' created.")
    else:
        print(f"data folder '{mouse}' already exists.")
    
    # Lists to hold percentages across sessions
    no_cr_percentages = []
    cr_percentages = []
    poor_cr_percentages = []
    
    # Dictionary to hold ISI counts for each session
    isi_data_per_session = []
    
    
    session_labels = []
    test_types = []
    trial_counts = []

    for i, session_date in enumerate(all_sessions):
        print(f'working on the session {session_date}')
        
        # Initialize default values for problematic sessions
        no_cr_count = 0
        cr_count = 0 
        poor_cr_count = 0
        isi_counts = {}
        session_error = False
        error_type = ""
        trial_count = 0
        
        try:
            trials = h5py.File(f"./data/beh/{mouse}/processed/{session_date}.h5")["trial_id"]

            fec, fec_time, trials = fec_zero(trials)
            
            cr_positive, CR_stat, CR_interval_avg, base_line_avg, cr_interval_idx, bl_interval_idx = CR_stat_indication(trials, fec, fec_time, static_threshold, AP_delay=10)
            
            all_id = []
            for trial_id in trials:
                all_id.append(trial_id)
            all_id.sort()

            trial_count = len(all_id)

            test_type = trials[all_id[0]]["test_type"][()]
            test_types.append(test_type)

            # Count CR types
            no_cr_count = sum(1 for id in all_id if CR_stat.get(id, 0) == 0)
            cr_count = sum(1 for id in all_id if CR_stat.get(id, 0) == 1)
            poor_cr_count = sum(1 for id in all_id if CR_stat.get(id, 0) == 2)
            
            # Count ISI values
            for trial_id in all_id:
                try:
                    trial = trials[trial_id]
                    isi_value = get_isi_value(trial)
                    grouped_isi = group_isi_value(isi_value)
                    
                    if grouped_isi in isi_counts:
                        isi_counts[grouped_isi] += 1
                    else:
                        isi_counts[grouped_isi] = 1
                except Exception as e:
                    # Add to 'ISI_Error' category for trials with ISI calculation errors
                    if 'ISI_Error' in isi_counts:
                        isi_counts['ISI_Error'] += 1
                    else:
                        isi_counts['ISI_Error'] = 1
           
            # Check if session appears empty after processing
            total = no_cr_count + cr_count + poor_cr_count
            if total == 0:
                session_error = True
                error_type = "Empty_Session"
                # Still include it but mark as error
                isi_counts = {'No_Data': 1}
                
        except Exception as e:
            print(f"Error processing session {session_date}: {e}")
            session_error = True
            error_type = "Processing_Error"
            # Set error indicators
            no_cr_count = 0
            cr_count = 0
            poor_cr_count = 1  # Show as 100% error in CR plot
            isi_counts = {'Processing_Error': 1}
            trial_count = 0
            breakpoint()

        total = max(no_cr_count + cr_count + poor_cr_count, 1)  # Avoid division by zero

        # Store CR percentages (always store, even for problematic sessions)
        no_cr_percentages.append((no_cr_count / total) * 100)
        cr_percentages.append((cr_count / total) * 100)
        poor_cr_percentages.append((poor_cr_count / total) * 100)
        
        # Store ISI data for this session
        isi_data_per_session.append(isi_counts)
        
        # Store trial count for this session
        trial_counts.append(trial_count)
        
        # Mark session with error in the label if there was a problem
        if session_error:
            session_labels.append(f"{session_date}*{error_type}")
        else:
            session_labels.append(session_date)

        print(f'session {session_date} added{"(with errors)" if session_error else ""}')

    # Convert to numpy arrays
    no_cr = np.array(no_cr_percentages)
    cr = np.array(cr_percentages)
    poor_cr = np.array(poor_cr_percentages)
    
    # Get all unique ISI values across all sessions for this mouse
    all_isi_values = set()
    for session_isi in isi_data_per_session:
        all_isi_values.update(session_isi.keys())
    all_isi_values = sorted(list(all_isi_values), key=lambda x: x if isinstance(x, (int, float)) else 999)
   
    # Create color map for ISI values (including error categories)
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_isi_values)))
    isi_color_map = dict(zip(all_isi_values, colors))
   
    # Use specific colors for error categories
    if 'Processing_Error' in isi_color_map:
        isi_color_map['Processing_Error'] = 'red'
    if 'No_Data' in isi_color_map:
        isi_color_map['No_Data'] = 'black'
    if 'ISI_Error' in isi_color_map:
        isi_color_map['ISI_Error'] = 'darkred'
    
   
    x = np.arange(len(session_labels))
    
    # CR Performance Plot (First row for this mouse)
    ax_cr = fig.add_subplot(gs[4 * mouse_i, 0:4])
    
    ax_cr.bar(x, cr, label='CR', color='red')
    ax_cr.bar(x, poor_cr, bottom=cr, label='Poor CR', color='purple')
    ax_cr.bar(x, no_cr, bottom=cr + poor_cr, label='No CR', color='blue')
    ax_cr.spines['top'].set_visible(False)
    ax_cr.spines['right'].set_visible(False)
    
    ax_cr.set_xticks(x)
    ax_cr.set_xticklabels(session_labels, rotation=75, ha='right', fontsize=5)
    ax_cr.set_ylim(0, 100)
    ax_cr.set_ylabel('CR Percentage (%)')
    ax_cr.set_title(f'{mouse} - CR Performance')
    ax_cr.legend(loc='upper right', fontsize=5)
    
    # ISI Distribution Plot (Second row for this mouse)
    ax_isi = fig.add_subplot(gs[4 * mouse_i + 1, 0:4])
    
    # Calculate percentages for each ISI value across sessions
    bottom_values = np.zeros(len(session_labels))
    
    for isi_value in all_isi_values:
        percentages = []
        for session_isi in isi_data_per_session:
            total_trials = sum(session_isi.values()) if session_isi else 1
            count = session_isi.get(isi_value, 0)
            percentages.append((count / total_trials) * 100)
        
        percentages = np.array(percentages)
        ax_isi.bar(x, percentages, bottom=bottom_values, 
                  label=f'ISI {isi_value}', color=isi_color_map[isi_value])
        bottom_values += percentages
    
    ax_isi.spines['top'].set_visible(False)
    ax_isi.spines['right'].set_visible(False)
    
    ax_isi.set_xticks(x)
    ax_isi.set_xticklabels(session_labels, rotation=75, ha='right', fontsize=5)
    ax_isi.set_ylim(0, 100)
    ax_isi.axhline(50, color = 'gray', alpha = 0.5, linestyle = '--')
    ax_isi.set_ylabel('ISI Percentage (%)')
    ax_isi.set_title(f'{mouse} - ISI Distribution')
    ax_isi.legend(loc='upper right', fontsize=4, ncol=2)
    
    # Test Type Distribution Plot (Third row for this mouse)
    ax_test = fig.add_subplot(gs[4 * mouse_i + 3, 0:4])

    # Color each point based on test type using viridis
    colors = plt.cm.hsv(np.array(test_types) / 4.0)
    ax_test.scatter(x, test_types, c=colors, s=50)
    ax_test.plot(x, test_types, '--', color='gray', alpha=0.5)
    
    ax_test.spines['top'].set_visible(False)
    ax_test.spines['right'].set_visible(False)
    
    ax_test.set_xticks(x)
    ax_test.set_xticklabels(session_labels, rotation=75, ha='right', fontsize=5)
    ax_test.set_ylabel('Test Type')
    ax_test.set_title(f'{mouse} - Test Type per Session')
    ax_test.grid(axis='y', color='gray', alpha=0.1)
    # Trial Count Plot (Fourth row for this mouse)
    ax_trials = fig.add_subplot(gs[4 * mouse_i + 2, 0:4])
    ax_test.set_ylim(-1, 5)
    ax_test.set_yticks(range(0, 5))
    ax_test.set_yticklabels(['0', '1', '2', '3', '4'])
    
    ax_trials.bar(x, trial_counts, color='green', alpha=0.7)
    ax_trials.spines['top'].set_visible(False)
    ax_trials.spines['right'].set_visible(False)
    
    ax_trials.set_xticks(x)
    ax_trials.set_xticklabels(session_labels, rotation=75, ha='right', fontsize=5)
    ax_trials.set_ylabel('Number of Trials')
    ax_trials.set_title(f'{mouse} - Trial Count per Session')

plt.tight_layout()
with PdfPages(mice_cr_performance_file) as pdf:
    pdf.savefig(fig)
    plt.close(fig)
