from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def run_isi_distribution_for_sessions(list_session_names, list_ops, list_session_output_path,
                                      read_trial_label):
    summaries = []
    for session_name, ops, output_path in zip(list_session_names, list_ops, list_session_output_path):
        print(f"\n--- ISI distribution {session_name} ---")
        try:
            summary = _plot_isi_distribution_for_session(
                session_name, ops, output_path, read_trial_label)
            print(summary)
        except Exception as exc:
            summary = {
                'session': session_name,
                'status': 'failed',
                'error': f'{type(exc).__name__}: {exc}',
            }
            print(summary)
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries)
    return summaries, summary_df


def _plot_isi_distribution_for_session(session_name, ops, output_path, read_trial_label):
    trial_labels_i = read_trial_label(ops)
    trial_type = np.asarray(trial_labels_i['trial_type'])
    isi = np.asarray(trial_labels_i['isi'], dtype=float)

    isi_short = isi[trial_type == 0]
    isi_long = isi[trial_type == 1]
    if len(isi_short) == 0 or len(isi_long) == 0:
        raise ValueError(
            f'Need both short and long trials; found short={len(isi_short)}, long={len(isi_long)}')

    fig, ax = plt.subplots(figsize=(8, 4))
    for values, label, color in [
        (isi_short, 'Short trials (normal)', 'blue'),
        (isi_long, 'Long trials (normal)', 'red'),
    ]:
        min_v, max_v, mean_v = values.min(), values.max(), values.mean()
        std_v = (max_v - min_v) / 6
        if std_v == 0:
            ax.axvline(mean_v, color=color, label=f'{label} (constant)')
            continue
        x = np.linspace(min_v, max_v, 200)
        y = norm.pdf(x, loc=mean_v, scale=std_v)
        if y.max() > 0:
            y = y / y.max()
        ax.plot(x, y, label=label, color=color)

    ax.axvline(x=0, color='black', linestyle='--', label='Stim1 onset')
    ax.set_xlim((-2000, 4000))
    ax.set_xlabel('Time (ms, relative to stim1 onset)')
    ax.set_ylabel('ISI Density')
    ax.set_title(f'ISI Distribution Relative to Stim1 Onset\n{session_name}')
    ax.legend()
    fig.tight_layout()

    figures_dir = Path(output_path) / 'Figures' / 'stim'
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / 'isi_distribution.png'
    pdf_path = figures_dir / 'isi_distribution.pdf'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return {
        'session': session_name,
        'status': 'ok',
        'n_short': int(len(isi_short)),
        'n_long': int(len(isi_long)),
        'short_mean': float(np.mean(isi_short)),
        'long_mean': float(np.mean(isi_long)),
        'short_min': float(np.min(isi_short)),
        'short_max': float(np.max(isi_short)),
        'long_min': float(np.min(isi_long)),
        'long_max': float(np.max(isi_long)),
        'png_path': str(png_path),
        'pdf_path': str(pdf_path),
        'error': None,
    }


def show_isi_distribution_viewer(isi_distribution_summaries):
    from IPython.display import HTML, display
    import ipywidgets as widgets

    ok_sessions = [row for row in isi_distribution_summaries if row.get('status') == 'ok']
    if not ok_sessions:
        display(HTML('<b>No successful ISI distribution outputs to display.</b>'))
        return

    def title(row):
        return (
            f"<h3 style='margin: 0 0 6px 0'>{row['session']}</h3>"
            f"<div>short={row.get('n_short')} | long={row.get('n_long')} | "
            f"short_mean={_fmt(row.get('short_mean'), 1)} | "
            f"long_mean={_fmt(row.get('long_mean'), 1)}</div>"
        )

    _show_png_sequence(ok_sessions, lambda row: row['png_path'], title)


def show_clustering_plot_viewer(clustering_summaries, list_session_names,
                                list_session_output_path):
    from IPython.display import HTML, display

    ok_sessions = [row for row in clustering_summaries if row.get('status') == 'ok']
    if not ok_sessions:
        display(HTML('<b>No successful clustering outputs to display.</b>'))
        return

    session_to_output = dict(zip(list_session_names, list_session_output_path))
    plot_options = {
        'Metrics': 'clustering_metrics.png',
        'Results': 'clustering_results.png',
    }

    def path_for(row, plot_key):
        return Path(session_to_output[row['session']]) / 'Figures' / 'clustering' / plot_options[plot_key]

    def title(row):
        return (
            f"<h3 style='margin: 0 0 6px 0'>{row['session']}</h3>"
            f"<div>status={row.get('status')} | neurons={row.get('n_neurons')} | "
            f"optimal_k={row.get('optimal_k')} | clusters={row.get('cluster_counts')}</div>"
        )

    _show_png_sequence(ok_sessions, path_for, title, plot_options=list(plot_options))


def show_decoding_plot_viewer(decoding_summaries, list_session_names,
                              list_session_output_path, decoding_mode):
    from IPython.display import HTML, display

    ok_sessions = [row for row in decoding_summaries if row.get('status') == 'ok']
    if not ok_sessions:
        display(HTML('<b>No successful decoding outputs to display.</b>'))
        return

    session_to_output = dict(zip(list_session_names, list_session_output_path))
    plot_options = {
        'Temporal accuracy': f'temporal_decoding_accuracy_{decoding_mode}.png',
        'Average weights': 'average_weights.png',
        'Confusion matrix': 'Confusion Matrix_Long_common.png',
        'Feature heatmap': 'SVM_Feature_Importance_Heatmap_Long_common.png',
        'Shuffled confusion matrix': 'Confusion Matrix_shuffle_Long_common.png',
        'Shuffled feature heatmap': 'SVM_Feature_Importance_Heatmap_shuffle_Long_common.png',
    }

    def path_for(row, plot_key):
        return Path(session_to_output[row['session']]) / 'Figures' / 'decoder' / plot_options[plot_key]

    def title(row):
        return (
            f"<h3 style='margin: 0 0 6px 0'>{row['session']}</h3>"
            f"<div>status={row.get('status')} | neurons={row.get('n_neurons')} | "
            f"trials={row.get('n_trials')} | cv={_fmt(row.get('best_cv_accuracy'))} | "
            f"test={_fmt(row.get('test_balanced_accuracy'))} | "
            f"peak temporal={_fmt(row.get('peak_temporal_accuracy'))}</div>"
        )

    _show_png_sequence(ok_sessions, path_for, title, plot_options=list(plot_options))


def show_session_plot_comparison(list_session_names, list_session_output_path):
    from IPython.display import HTML, Image, clear_output, display
    import ipywidgets as widgets

    if not list_session_names or not list_session_output_path:
        raise RuntimeError('Session names/output directories are not defined. Run setup cells first.')
    if len(list_session_names) != len(list_session_output_path):
        raise RuntimeError('Session name and output directory counts do not match.')

    figs_by_session = {}
    for name, out_dir in zip(list_session_names, list_session_output_path):
        fig_dir = Path(out_dir) / 'Figures'
        files = []
        if fig_dir.exists():
            files = sorted(
                str(fp.relative_to(fig_dir))
                for fp in fig_dir.rglob('*')
                if fp.is_file() and fp.suffix.lower() == '.png'
            )
        figs_by_session[name] = {'base': fig_dir, 'files': files}

    common_files = None
    for name in list_session_names:
        fset = set(figs_by_session[name]['files'])
        common_files = fset if common_files is None else (common_files & fset)
    common_files = sorted(common_files or [])
    if not common_files:
        raise RuntimeError('No common PNG figure files found across selected sessions under Figures/.')

    left_dropdown = widgets.Dropdown(
        options=list_session_names, value=list_session_names[0],
        description='Left', layout=widgets.Layout(width='48%'))
    right_dropdown = widgets.Dropdown(
        options=list_session_names, value=list_session_names[min(1, len(list_session_names) - 1)],
        description='Right', layout=widgets.Layout(width='48%'))
    plot_dropdown = widgets.Dropdown(
        options=common_files, value=common_files[0],
        description='Plot', layout=widgets.Layout(width='95%'))

    left_name = widgets.HTML()
    right_name = widgets.HTML()
    left_out = widgets.Output(layout=widgets.Layout(border='1px solid #ddd', padding='6px'))
    right_out = widgets.Output(layout=widgets.Layout(border='1px solid #ddd', padding='6px'))

    def show_plot(out_widget, session_name, rel_path):
        with out_widget:
            clear_output(wait=True)
            fpath = figs_by_session[session_name]['base'] / rel_path
            display(HTML(f"<b>{session_name}</b><br><code>{fpath}</code>"))
            display(Image(filename=str(fpath)))

    def update(*_):
        left_session = left_dropdown.value
        right_session = right_dropdown.value
        rel = plot_dropdown.value
        left_name.value = f"<b>{left_session}</b>"
        right_name.value = f"<b>{right_session}</b>"
        show_plot(left_out, left_session, rel)
        show_plot(right_out, right_session, rel)

    left_dropdown.observe(update, names='value')
    right_dropdown.observe(update, names='value')
    plot_dropdown.observe(update, names='value')

    controls = widgets.VBox([
        plot_dropdown,
        widgets.HBox([left_dropdown, right_dropdown],
                     layout=widgets.Layout(width='100%', justify_content='space-between')),
    ])
    panel = widgets.HBox([
        widgets.VBox([left_name, left_out], layout=widgets.Layout(width='50%')),
        widgets.VBox([right_name, right_out], layout=widgets.Layout(width='50%')),
    ], layout=widgets.Layout(width='100%'))

    display(controls)
    display(panel)
    update()


def show_raw_dff_trace_viewer(list_session_names, list_ops, read_dff,
                              max_points=10000, max_all_neurons=80):
    from IPython.display import HTML, display
    import ipywidgets as widgets

    if not list_session_names or not list_ops:
        raise RuntimeError('Session names/list_ops are not defined. Run setup cells first.')

    cache = {}
    session_dropdown = widgets.Dropdown(
        options=list_session_names,
        value=list_session_names[0],
        description='Session',
        layout=widgets.Layout(width='70%'),
    )
    mode_dropdown = widgets.Dropdown(
        options=['All neurons (offset)', 'Single neuron'],
        value='All neurons (offset)',
        description='Mode',
        layout=widgets.Layout(width='35%'),
    )
    neuron_dropdown = widgets.Dropdown(
        options=[],
        description='ROI',
        layout=widgets.Layout(width='30%'),
    )
    frame_start = widgets.IntText(value=0, description='Start')
    frame_window = widgets.IntText(value=max_points, description='Window')
    message = widgets.HTML()
    image = widgets.Image(format='png')

    def load_session(session_name):
        if session_name not in cache:
            idx = list_session_names.index(session_name)
            ops = list_ops[idx]
            root_dff_path = Path(ops['save_path0']) / 'dff.h5'
            if root_dff_path.exists():
                import h5py
                with h5py.File(root_dff_path, 'r') as f:
                    dff = np.asarray(f['dff'])
            else:
                dff = np.asarray(read_dff(ops))
            cache[session_name] = dff
        return cache[session_name]

    def downsample_indices(n_time):
        start = max(0, int(frame_start.value or 0))
        window = max(1, int(frame_window.value or max_points))
        stop = min(n_time, start + window)
        step = max(1, int(np.ceil((stop - start) / max_points)))
        return np.arange(start, stop, step), step

    def update_neuron_options(*_):
        dff = load_session(session_dropdown.value)
        neuron_dropdown.options = list(range(dff.shape[0]))
        neuron_dropdown.value = 0 if dff.shape[0] > 0 else None

    def render(*_):
        session_name = session_dropdown.value
        dff = load_session(session_name)
        n_neurons, n_time = dff.shape
        t_idx, step = downsample_indices(n_time)
        x = t_idx

        message.value = (
            f"<b>{session_name}</b> | neurons={n_neurons} | "
            f"timepoints={n_time} | frames={int(t_idx[0]) if len(t_idx) else 0}-"
            f"{int(t_idx[-1]) if len(t_idx) else 0} | plotted every {step} frame(s)"
        )
        fig, ax = plt.subplots(figsize=(14, 7))

        if mode_dropdown.value == 'Single neuron':
            neuron = int(neuron_dropdown.value or 0)
            ax.plot(x, dff[neuron, t_idx], linewidth=0.8)
            ax.set_ylabel('dF/F')
            ax.set_title(f'Raw DFF trace: {session_name} | ROI {neuron}')
        else:
            n_plot = min(n_neurons, max_all_neurons)
            sample_neurons = np.linspace(0, n_neurons - 1, n_plot, dtype=int)
            sample = dff[sample_neurons][:, t_idx]
            ptp = np.nanpercentile(sample, 99) - np.nanpercentile(sample, 1)
            offset = ptp if np.isfinite(ptp) and ptp > 0 else 1.0
            for plot_i, neuron in enumerate(sample_neurons):
                ax.plot(x, sample[plot_i] + plot_i * offset, linewidth=0.5)
            ax.set_ylabel(f'dF/F + offset ({n_plot}/{n_neurons} neurons)')
            ax.set_title(f'Raw DFF traces: {session_name}')

        ax.set_xlabel('Frame')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
        plt.close(fig)
        image.value = buf.getvalue()

    session_dropdown.observe(lambda change: (update_neuron_options(), render()), names='value')
    mode_dropdown.observe(render, names='value')
    neuron_dropdown.observe(render, names='value')
    frame_start.observe(render, names='value')
    frame_window.observe(render, names='value')

    update_neuron_options()
    display(widgets.VBox([
        session_dropdown,
        widgets.HBox([mode_dropdown, neuron_dropdown]),
        widgets.HBox([frame_start, frame_window]),
        message,
        image,
    ]))
    render()


def show_fov_dff_click_viewer(list_session_names, list_ops, read_masks, read_dff,
                              max_points=10000):
    from IPython.display import display
    import ipywidgets as widgets
    import plotly.graph_objects as go

    if not list_session_names or not list_ops:
        raise RuntimeError('Session names/list_ops are not defined. Run setup cells first.')

    cache = {}
    state = {'fov_fig': None, 'trace_fig': None}
    session_dropdown = widgets.Dropdown(
        options=list_session_names,
        value=list_session_names[0],
        description='Session',
        layout=widgets.Layout(width='70%'),
    )
    neuron_dropdown = widgets.Dropdown(
        options=[],
        description='ROI',
        layout=widgets.Layout(width='35%'),
    )
    frame_start = widgets.IntText(value=0, description='Start')
    frame_window = widgets.IntText(value=max_points, description='Window')
    status = widgets.HTML()
    legend = widgets.HTML(
        value=(
            "<b>Mask legend</b>: "
            "<span style='color:#0000ff'>blue</span> = excitatory, "
            "<span style='color:#ff00ff'>pink</span> = inhibitory, "
            "<span style='color:#808080'>white</span> = unsure"
        )
    )
    fov_output = widgets.Output()
    trace_output = widgets.Output()

    def load_session(session_name):
        if session_name not in cache:
            idx = list_session_names.index(session_name)
            ops = list_ops[idx]
            labels, masks, mean_func, max_func, mean_anat, masks_anat = read_masks(ops)
            root_dff_path = Path(ops['save_path0']) / 'dff.h5'
            neural_trials_path = Path(ops['save_path0']) / 'neural_trials.h5'
            if root_dff_path.exists():
                import h5py
                with h5py.File(root_dff_path, 'r') as f:
                    dff = np.asarray(f['dff'])
            else:
                dff = np.asarray(read_dff(ops))
            time_ms = _load_dff_time_ms(neural_trials_path, dff.shape[1], ops)
            centroids, roi_ids = _roi_centroids(np.asarray(masks))
            roi_to_neuron = _build_roi_id_to_neuron_map(
                roi_ids=roi_ids,
                n_dff=dff.shape[0],
                labels=np.asarray(labels),
            )
            centroid_neurons = np.array(
                [roi_to_neuron.get(int(roi_id), -1) for roi_id in roi_ids],
                dtype=int,
            )
            neuron_to_centroid = {
                int(neuron): idx
                for idx, neuron in enumerate(centroid_neurons)
                if neuron >= 0
            }
            neuron_to_roi = {
                int(neuron): int(roi_id)
                for roi_id, neuron in roi_to_neuron.items()
                if neuron >= 0
            }
            cache[session_name] = {
                'labels': np.asarray(labels),
                'masks': np.asarray(masks),
                'mean_func': np.asarray(mean_func),
                'fov_overlay': _superimpose_mask_func_image(
                    np.asarray(mean_func),
                    np.asarray(masks),
                    np.asarray(labels),
                ),
                'dff': dff,
                'time_ms': time_ms,
                'centroids': centroids,
                'roi_ids': roi_ids,
                'roi_to_neuron': roi_to_neuron,
                'centroid_neurons': centroid_neurons,
                'neuron_to_centroid': neuron_to_centroid,
                'neuron_to_roi': neuron_to_roi,
            }
        return cache[session_name]

    def downsample_indices(n_time):
        start = max(0, int(frame_start.value or 0))
        window = max(1, int(frame_window.value or max_points))
        stop = min(n_time, start + window)
        step = max(1, int(np.ceil((stop - start) / max_points)))
        return np.arange(start, stop, step), step

    def update_neuron_options():
        session = load_session(session_dropdown.value)
        n_dff = session['dff'].shape[0]
        options = []
        for neuron in range(n_dff):
            roi_label = ''
            roi_id = session['neuron_to_roi'].get(neuron)
            if roi_id is not None:
                roi_label = f' / ROI {roi_id}'
            options.append((f'ROI {neuron}{roi_label}', neuron))
        neuron_dropdown.options = options
        neuron_dropdown.value = 0 if n_dff > 0 else None

    def selected_centroid(session, neuron):
        if neuron is None:
            return [], []
        centroid_idx = session['neuron_to_centroid'].get(int(neuron))
        if centroid_idx is None or centroid_idx >= len(session['centroids']):
            return [], []
        return [session['centroids'][centroid_idx, 0]], [session['centroids'][centroid_idx, 1]]

    def make_fov_figure():
        session_name = session_dropdown.value
        session = load_session(session_name)
        neuron = int(neuron_dropdown.value or 0)
        xs = session['centroids'][:, 0] if len(session['centroids']) else []
        ys = session['centroids'][:, 1] if len(session['centroids']) else []
        customdata = session['centroid_neurons'] if len(session['centroids']) else []
        sel_x, sel_y = selected_centroid(session, neuron)

        fig = go.FigureWidget()
        fig.add_trace(go.Image(z=session['fov_overlay'], name='FOV'))
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='markers',
            name='ROIs',
            customdata=customdata,
            marker=dict(size=8, color='cyan', symbol='circle-open', line=dict(width=1.5)),
            hovertemplate='ROI %{customdata}<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=sel_x,
            y=sel_y,
            mode='markers',
            name='Selected',
            marker=dict(size=16, color='yellow', symbol='circle-open', line=dict(width=3)),
            hoverinfo='skip',
        ))
        fig.update_layout(
            title=f"{session_name} | click an ROI centroid | {session['dff'].shape[0]} DFF traces",
            width=720,
            height=720,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False,
        )
        fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
        fig.update_yaxes(showgrid=False, zeroline=False, visible=False, scaleanchor='x', autorange='reversed')
        fig.data[1].on_click(on_roi_click)
        return fig

    def make_trace_figure():
        session_name = session_dropdown.value
        session = load_session(session_name)
        dff = session['dff']
        time_ms = session['time_ms']
        neuron = int(neuron_dropdown.value or 0)
        t_idx, step = downsample_indices(dff.shape[1])
        x_ms = time_ms[t_idx] if len(t_idx) else []
        y = dff[neuron, t_idx] if len(t_idx) else []
        fig = go.FigureWidget()
        fig.add_trace(go.Scatter(x=x_ms, y=y, mode='lines', line=dict(width=1), name='dF/F'))
        fig.update_layout(
            title=f'{session_name} | ROI {neuron}',
            width=1000,
            height=320,
            margin=dict(l=60, r=20, t=50, b=50),
            xaxis_title='Time (ms)',
            yaxis_title='dF/F',
        )
        status.value = (
            f"<b>{session_name}</b> | click an ROI centroid or use dropdown | ROI={neuron} | "
            f"time={_fmt_ms(x_ms[0]) if len(x_ms) else '0.0 ms'}-"
            f"{_fmt_ms(x_ms[-1]) if len(x_ms) else '0.0 ms'} | every {step} frame(s)"
        )
        return fig

    def refresh_fov_selection():
        fig = state['fov_fig']
        if fig is None:
            return
        session = load_session(session_dropdown.value)
        neuron = int(neuron_dropdown.value or 0)
        sel_x, sel_y = selected_centroid(session, neuron)
        with fig.batch_update():
            fig.data[2].x = sel_x
            fig.data[2].y = sel_y

    def refresh_trace():
        fig = state['trace_fig']
        if fig is None:
            return
        session_name = session_dropdown.value
        session = load_session(session_name)
        dff = session['dff']
        time_ms = session['time_ms']
        neuron = int(neuron_dropdown.value or 0)
        t_idx, step = downsample_indices(dff.shape[1])
        x_ms = time_ms[t_idx] if len(t_idx) else []
        y = dff[neuron, t_idx] if len(t_idx) else []
        with fig.batch_update():
            fig.data[0].x = x_ms
            fig.data[0].y = y
            fig.layout.title = f'{session_name} | ROI {neuron}'
        status.value = (
            f"<b>{session_name}</b> | click an ROI centroid or use dropdown | ROI={neuron} | "
            f"time={_fmt_ms(x_ms[0]) if len(x_ms) else '0.0 ms'}-"
            f"{_fmt_ms(x_ms[-1]) if len(x_ms) else '0.0 ms'} | every {step} frame(s)"
        )

    def select_neuron(neuron):
        neuron = int(neuron)
        if neuron_dropdown.value != neuron:
            neuron_dropdown.value = neuron
        else:
            refresh_fov_selection()
            refresh_trace()

    def on_roi_click(trace, points, selector):
        if not points.point_inds:
            return
        point_idx = int(points.point_inds[0])
        session = load_session(session_dropdown.value)
        if point_idx >= len(session['roi_ids']):
            return
        roi_id = int(session['roi_ids'][point_idx])
        neuron = int(session['centroid_neurons'][point_idx])
        if neuron < 0 or neuron >= session['dff'].shape[0]:
            status.value = f'<b>ROI {roi_id}</b> has no valid dF/F row mapping'
            return
        status.value = f'<b>Selected ROI {roi_id}</b> mapped to ROI {neuron}'
        select_neuron(neuron)

    def rebuild_figures():
        status.value = '<b>Loading Plotly FOV/DFF viewer...</b>'
        try:
            state['fov_fig'] = make_fov_figure()
            state['trace_fig'] = make_trace_figure()
            with fov_output:
                fov_output.clear_output(wait=True)
                display(state['fov_fig'])
            with trace_output:
                trace_output.clear_output(wait=True)
                display(state['trace_fig'])
        except Exception as exc:
            state['fov_fig'] = None
            state['trace_fig'] = None
            with fov_output:
                fov_output.clear_output(wait=True)
            with trace_output:
                trace_output.clear_output(wait=True)
            status.value = f'<b>Viewer render failed:</b> {type(exc).__name__}: {exc}'

    def on_session_change(*_):
        update_neuron_options()
        rebuild_figures()

    def on_neuron_change(*_):
        refresh_fov_selection()
        refresh_trace()

    session_dropdown.observe(lambda change: on_session_change(), names='value')
    neuron_dropdown.observe(lambda change: on_neuron_change(), names='value')
    frame_start.observe(lambda change: refresh_trace(), names='value')
    frame_window.observe(lambda change: refresh_trace(), names='value')

    display(widgets.VBox([
        session_dropdown,
        widgets.HBox([neuron_dropdown, frame_start, frame_window]),
        status,
        legend,
        fov_output,
        trace_output,
    ]))
    update_neuron_options()
    rebuild_figures()

def _roi_centroids(masks):
    roi_ids = np.array([roi_id for roi_id in np.unique(masks) if roi_id > 0], dtype=int)
    centroids = []
    for roi_id in roi_ids:
        yy, xx = np.where(masks == roi_id)
        if len(xx) == 0:
            continue
        centroids.append((float(np.mean(xx)), float(np.mean(yy))))
    return np.asarray(centroids), roi_ids[:len(centroids)]


def _clicked_roi_id(masks, centroids, roi_ids, x, y):
    yi = int(round(y))
    xi = int(round(x))
    if 0 <= yi < masks.shape[0] and 0 <= xi < masks.shape[1]:
        roi_id = int(masks[yi, xi])
        if roi_id > 0:
            return roi_id
    if len(centroids) == 0:
        return None
    distances = np.sqrt((centroids[:, 0] - x) ** 2 + (centroids[:, 1] - y) ** 2)
    nearest = int(np.argmin(distances))
    return int(roi_ids[nearest]) if distances[nearest] <= 12 else None


def _roi_id_to_dff_index(roi_id, n_dff, roi_ids):
    if 0 <= roi_id < n_dff:
        return int(roi_id)
    if 0 <= roi_id - 1 < n_dff:
        return int(roi_id - 1)
    matches = np.where(roi_ids == roi_id)[0]
    if len(matches) and matches[0] < n_dff:
        return int(matches[0])
    return int(np.clip(roi_id, 0, n_dff - 1))


def _build_roi_id_to_neuron_map(roi_ids, n_dff, labels=None):
    roi_ids = np.asarray(roi_ids, dtype=int)
    labels = np.asarray(labels) if labels is not None else None

    if n_dff <= 0 or len(roi_ids) == 0:
        return {}

    has_background_label = labels is not None and labels.ndim == 1 and len(labels) == (n_dff + 1)
    if has_background_label:
        return {
            int(roi_id): int(roi_id - 1)
            for roi_id in roi_ids
            if 1 <= roi_id <= n_dff
        }

    if labels is not None and labels.ndim == 1 and len(labels) == n_dff:
        return {
            int(roi_id): int(roi_id)
            for roi_id in roi_ids
            if 0 <= roi_id < n_dff
        }

    if len(roi_ids) == n_dff:
        return {int(roi_id): idx for idx, roi_id in enumerate(roi_ids)}

    mapping = {}
    for idx, roi_id in enumerate(roi_ids):
        if idx >= n_dff:
            break
        mapping[int(roi_id)] = idx
    return mapping


def _load_dff_time_ms(neural_trials_path, n_time, ops):
    if neural_trials_path.exists():
        import h5py
        with h5py.File(neural_trials_path, 'r') as f:
            if 'neural_trials' in f and 'time' in f['neural_trials']:
                time_ms = np.asarray(f['neural_trials']['time'], dtype=float).reshape(-1)
                if len(time_ms) == n_time:
                    return time_ms - time_ms[0]

    fs = float(ops.get('fs', 30.0) or 30.0)
    dt_ms = 1000.0 / fs
    return np.arange(n_time, dtype=float) * dt_ms


def _superimpose_mask_func_image(mean_func, masks, labels):
    base = (_normalize_image(mean_func) * 255).astype(np.uint8)
    rgb = np.zeros((*base.shape, 3), dtype=np.uint8)
    rgb[..., 1] = base

    masks = np.asarray(masks)
    labels = np.asarray(labels).reshape(-1)
    roi_ids = [int(roi_id) for roi_id in np.unique(masks) if roi_id > 0]
    if not roi_ids:
        return rgb

    for roi_id in roi_ids:
        roi_mask = masks == roi_id
        if not np.any(roi_mask):
            continue
        boundary = _mask_boundary(roi_mask)
        if not np.any(boundary):
            continue
        rgb[boundary] = _roi_boundary_color(_roi_label_for_id(roi_id, labels))

    return rgb


def _mask_boundary(mask):
    mask = np.asarray(mask, dtype=bool)
    up = np.zeros_like(mask)
    down = np.zeros_like(mask)
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)
    up[1:, :] = mask[:-1, :]
    down[:-1, :] = mask[1:, :]
    left[:, 1:] = mask[:, :-1]
    right[:, :-1] = mask[:, 1:]
    interior = mask & up & down & left & right
    return mask & ~interior


def _roi_label_for_id(roi_id, labels):
    roi_id = int(roi_id)
    if labels.ndim != 1 or len(labels) == 0:
        return 0
    if 0 <= roi_id < len(labels):
        return int(labels[roi_id])
    if 0 <= roi_id - 1 < len(labels):
        return int(labels[roi_id - 1])
    return 0


def _roi_boundary_color(label):
    if label == 1:
        return np.array([255, 0, 255], dtype=np.uint8)
    if label == -1:
        return np.array([0, 0, 255], dtype=np.uint8)
    return np.array([255, 255, 255], dtype=np.uint8)


def _normalize_image(image):
    arr = np.asarray(image, dtype=float)
    low, high = np.nanpercentile(arr, [1, 99])
    return np.clip((arr - low) / (high - low + 1e-9), 0, 1)


def _show_png_sequence(rows, path_for, title_for, plot_options=None):
    from IPython.display import HTML, display
    import ipywidgets as widgets

    idx = {'value': 0}
    title = widgets.HTML()
    prev = widgets.Button(description='< Prev')
    next_ = widgets.Button(description='Next >')
    image = widgets.Image(format='png')
    message = widgets.HTML()

    if plot_options is None:
        plot_dropdown = None
    else:
        plot_dropdown = widgets.Dropdown(
            options=plot_options, value=plot_options[0], description='Plot')

    def selected_path(row):
        if plot_dropdown is None:
            return Path(path_for(row))
        return Path(path_for(row, plot_dropdown.value))

    def render():
        row = rows[idx['value']]
        image_path = selected_path(row)
        title.value = title_for(row)
        if image_path.exists():
            image.value = image_path.read_bytes()
            message.value = f"<code>{image_path}</code>"
        else:
            image.value = b''
            message.value = f"<b>Missing output:</b> <code>{image_path}</code>"

    def go_prev(_):
        idx['value'] = max(0, idx['value'] - 1)
        render()

    def go_next(_):
        idx['value'] = min(len(rows) - 1, idx['value'] + 1)
        render()

    prev.on_click(go_prev)
    next_.on_click(go_next)
    controls = [prev, next_]
    if plot_dropdown is not None:
        plot_dropdown.observe(lambda change: render(), names='value')
        controls.append(plot_dropdown)

    display(title)
    display(widgets.HBox(controls))
    display(message)
    display(image)
    render()


def _fmt(value, digits=3):
    if value is None:
        return 'NA'
    try:
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return 'NA'


def _fmt_ms(value, digits=1):
    try:
        return f'{float(value):.{digits}f} ms'
    except (TypeError, ValueError):
        return 'NA'
