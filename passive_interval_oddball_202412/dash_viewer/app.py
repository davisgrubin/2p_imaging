from __future__ import annotations

import itertools
from datetime import datetime
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb
from dash import Dash, Input, Output, State, dcc, html, callback_context

from .data import (
    BLOCK_LABELS,
    CELL_SUBSET_ORDER,
    DataRepository,
    aggregate_counts,
    aggregate_timecourse,
    normalise_subset_label,
    compute_selection_summary,
)

AGGREGATION_FIELDS: Dict[str, str] = {
    "region": "region",
    "subject": "subject",
    "session": "session",
    "neuron_type": "cell_subset",
}


def _options(values: Sequence[str]) -> List[Dict[str, str]]:
    return [{"label": value, "value": value} for value in values]


def _subset_options(values: Sequence[str]) -> List[Dict[str, str]]:
    return [{"label": normalise_subset_label(value), "value": value} for value in values]


def _format_timestamp(ts: float | None) -> str:
    if ts is None:
        return "Data not loaded"
    return f"Loaded {datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')}"


def _build_counts_table(records: List[Dict[str, Any]], group_field: str) -> html.Div:
    if not records:
        return html.Div("No matching summary rows for this selection.", className="counts-empty")

    header_labels = {
        "label": group_field.replace("_", " ").title(),
        "n_sessions": "# Sessions",
        "n_subjects": "# Subjects",
        "n_trials": "# Trials",
        "n_trials_short": "Trials (Short)",
        "n_trials_long": "Trials (Long)",
        "n_neurons_sampled": "Sampled neurons",
        "n_neurons_total": "Total neurons",
        "n_neuron_types": "Neuron types",
    }
    columns = [key for key in header_labels if key != "label"]

    header_row = [html.Th(header_labels["label"])] + [html.Th(header_labels[col]) for col in columns]
    body_rows = []
    for record in records:
        cells = [html.Td(record["label"])]
        for col in columns:
            value = record.get(col)
            if value is None:
                cells.append(html.Td("—"))
            else:
                cells.append(html.Td(f"{int(value):,}" if isinstance(value, (int, float)) else value))
        body_rows.append(html.Tr(cells))

    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr(header_row)),
                html.Tbody(body_rows),
            ],
            className="counts-table",
        )
    )


def _prepare_count_records(df, group_field: str) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        value = row[group_field]
        if group_field == "cell_subset":
            label = normalise_subset_label(value)
        else:
            label = str(value)
        record = {"label": label}
        for key in [
            "n_sessions",
            "n_subjects",
            "n_trials",
            "n_trials_short",
            "n_trials_long",
            "n_neurons_sampled",
            "n_neurons_total",
            "n_neuron_types",
        ]:
            if key in row and not (row[key] != row[key]):  # NaN check
                record[key] = row[key]
        records.append(record)
    return records


def _color_cycle() -> Sequence[str]:
    palette = (
        go.Figure().layout.colorway
        or ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
    )
    return itertools.cycle(palette)


def _with_alpha(color: str, alpha: float) -> str:
    if not isinstance(color, str):
        return color
    color = color.strip()
    if color.startswith("#"):
        r, g, b = hex_to_rgb(color)
        return f"rgba({r},{g},{b},{alpha})"
    match = re.match(r"rgba?\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)", color)
    if match:
        r, g, b = match.groups()
        return f"rgba({r},{g},{b},{alpha})"
    return color


def _add_sem_band(
    fig: go.Figure,
    *,
    x,
    mean,
    sem,
    color: str,
    legend_group: str,
    opacity: float = 0.18,
) -> None:
    if sem is None:
        return
    x_arr = np.asarray(x, dtype=float)
    mean_arr = np.asarray(mean, dtype=float)
    sem_arr = np.asarray(sem, dtype=float)
    if x_arr.size == 0 or mean_arr.size == 0 or sem_arr.size == 0:
        return
    finite_mask = np.isfinite(sem_arr)
    if not finite_mask.any():
        return
    sem_arr = np.where(finite_mask, sem_arr, 0.0)
    if np.allclose(sem_arr, 0.0):
        return
    upper = (mean_arr + sem_arr).tolist()
    lower = (mean_arr - sem_arr).tolist()
    rgba_color = _with_alpha(color, opacity)
    fig.add_trace(
        go.Scatter(
            x=x_arr.tolist(),
            y=lower,
            mode="lines",
            line=dict(color=color, width=0),
            legendgroup=legend_group,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_arr.tolist(),
            y=upper,
            mode="lines",
            line=dict(color=color, width=0),
            fill="tonexty",
            fillcolor=rgba_color,
            legendgroup=legend_group,
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _format_summary_annotation(summary: Dict[str, int]) -> str:
    if not summary:
        return ""
    labels = {
        "n_sessions": "n sessions",
        "n_subjects": "n subjects",
        "n_neurons_total": "total neurons",
        "n_neuron_types": "neuron types",
    }
    order = ["n_sessions", "n_subjects", "n_neurons_total", "n_neuron_types"]
    lines = []
    for key in order:
        value = summary.get(key)
        if value:
            lines.append(f"{labels[key]}: {int(value)}")
    return "<br>".join(lines)


def _build_timecourse_figure(
    df,
    *,
    group_field: str,
    y_mode: str,
    y_range: Sequence[float],
    summary_info: Dict[str, int],
) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template="seaborn")
        fig.add_annotation(
            text="No data for this selection.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_xaxes(title_text="Time relative to stimulus (ms)")
        fig.update_yaxes(title_text="Decoder accuracy")
        return fig

    df = df.sort_values(["block_type", "time_center_ms"])
    palette = _color_cycle()

    for group_value in df[group_field].dropna().unique():
        group_label = normalise_subset_label(group_value) if group_field == "cell_subset" else str(group_value)
        color = next(palette)
        group_df = df[df[group_field] == group_value]
        for block_type in group_df["block_type"].dropna().astype(int).unique():
            block_label = BLOCK_LABELS.get(block_type, f"Block {block_type}")
            legend_group = f"{group_label}-{block_type}"
            block_df = group_df[group_df["block_type"] == block_type]
            block_df = block_df.sort_values("time_center_ms")
            if block_df.empty:
                continue
            _add_sem_band(
                fig,
                x=block_df["time_center_ms"],
                mean=block_df["acc_mean"],
                sem=block_df["acc_sem"],
                color=color,
                legend_group=legend_group,
            )
            fig.add_trace(
                go.Scatter(
                    x=block_df["time_center_ms"],
                    y=block_df["acc_mean"],
                    mode="lines",
                    name=f"{group_label} — {block_label}",
                    legendgroup=legend_group,
                    line=dict(color=color, width=2),
                    hovertemplate="Time: %{x:.0f} ms<br>Accuracy: %{y:.3f}<extra>%{fullData.name}</extra>",
                )
            )
            _add_sem_band(
                fig,
                x=block_df["time_center_ms"],
                mean=block_df["chance_mean"],
                sem=block_df.get("chance_sem"),
                color=color,
                legend_group=f"{legend_group}-chance",
                opacity=0.12,
            )
            fig.add_trace(
                go.Scatter(
                    x=block_df["time_center_ms"],
                    y=block_df["chance_mean"],
                    mode="lines",
                    name=f"{group_label} — {block_label} (chance)",
                    legendgroup=legend_group,
                    line=dict(color=color, dash="dot"),
                    hovertemplate="Time: %{x:.0f} ms<br>Chance: %{y:.3f}<extra>%{fullData.name}</extra>",
                    showlegend=False,
                )
            )

    fig.update_layout(
        template="seaborn",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50, r=20, b=80, l=60),
        xaxis_title="Time relative to stimulus (ms)",
        yaxis_title="Decoder accuracy",
    )

    if y_mode == "manual" and y_range and len(y_range) == 2:
        lower, upper = sorted(y_range)
        fig.update_yaxes(range=[lower, upper])
    else:
        fig.update_yaxes(autorange=True)

    summary_text = _format_summary_annotation(summary_info)
    if summary_text:
        fig.add_annotation(
            text=summary_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            align="left",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0)",
            borderpad=6,
        )

    return fig


def create_dash_app(data_root: Path) -> Dash:
    repository = DataRepository(data_root)
    repository.reload()

    regions = repository.available_regions()
    subsets = repository.available_subsets()
    targets = repository.available_targets()

    default_target = "change_vs_repeat" if "change_vs_repeat" in targets else (targets[0] if targets else None)
    default_regions = regions
    default_subsets = subsets or CELL_SUBSET_ORDER
    default_subjects = repository.subjects_for_regions(default_regions)
    default_sessions = repository.sessions_for_selection(default_regions, default_subjects)

    app = Dash(__name__)
    app.title = "Dash Interactive Decoding Viewer"

    app.layout = html.Div(
        [
            html.H1("Dash Interactive Decoding Viewer"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Target decoder"),
                            dcc.Dropdown(
                                id="target-filter",
                                options=_options(targets),
                                value=default_target,
                                clearable=False,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Aggregate by"),
                            dcc.RadioItems(
                                id="aggregator-radio",
                                options=[
                                    {"label": "Region", "value": "region"},
                                    {"label": "Subject", "value": "subject"},
                                    {"label": "Session", "value": "session"},
                                    {"label": "Neuron type", "value": "neuron_type"},
                                ],
                                value="region",
                                labelStyle={"display": "inline-block", "margin-right": "12px"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Regions"),
                            dcc.Dropdown(
                                id="region-filter",
                                options=_options(regions),
                                value=default_regions,
                                multi=True,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Subjects"),
                            dcc.Dropdown(
                                id="subject-filter",
                                options=_options(default_subjects),
                                value=default_subjects,
                                multi=True,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Sessions"),
                            dcc.Dropdown(
                                id="session-filter",
                                options=_options(default_sessions),
                                value=[],
                                multi=True,
                                placeholder="All sessions",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Neuron subsets"),
                            dcc.Dropdown(
                                id="subset-filter",
                                options=_subset_options(subsets or CELL_SUBSET_ORDER),
                                value=default_subsets,
                                multi=True,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Block types"),
                            dcc.Checklist(
                                id="block-type-filter",
                                options=[
                                    {"label": BLOCK_LABELS[0], "value": 0},
                                    {"label": BLOCK_LABELS[1], "value": 1},
                                ],
                                value=[0, 1],
                                labelStyle={"display": "inline-block", "margin-right": "12px"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Y-axis mode"),
                            dcc.RadioItems(
                                id="y-axis-mode",
                                options=[
                                    {"label": "Auto", "value": "auto"},
                                    {"label": "Manual", "value": "manual"},
                                ],
                                value="auto",
                                labelStyle={"display": "inline-block", "margin-right": "12px"},
                            ),
                            dcc.RangeSlider(
                                id="y-axis-range",
                                min=0.0,
                                max=1.0,
                                value=[0.4, 1.0],
                                step=0.01,
                                tooltip={"placement": "bottom"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Button("Reload data", id="reload-data", n_clicks=0),
                            html.Span(id="last-loaded-label", style={"margin-left": "12px"}),
                        ],
                        className="reload-row",
                    ),
                ],
                className="controls-grid",
            ),
            dcc.Graph(id="timecourse-graph", config={"displaylogo": False, "scrollZoom": True}),
            html.Div(id="counts-container"),
        ],
        className="viewer-root",
    )

    @app.callback(
        Output("subject-filter", "options"),
        Output("subject-filter", "value"),
        Input("region-filter", "value"),
        State("subject-filter", "value"),
    )
    def update_subjects(regions_selected, current_subjects):
        regions_selected = regions_selected or []
        current_subjects = current_subjects or []
        candidates = repository.subjects_for_regions(regions_selected)
        updated_value = [subject for subject in current_subjects if subject in candidates]
        if not updated_value:
            updated_value = candidates
        return _options(candidates), updated_value

    @app.callback(
        Output("session-filter", "options"),
        Output("session-filter", "value"),
        Input("region-filter", "value"),
        Input("subject-filter", "value"),
        State("session-filter", "value"),
    )
    def update_sessions(regions_selected, subjects_selected, current_sessions):
        regions_selected = regions_selected or []
        subjects_selected = subjects_selected or []
        current_sessions = current_sessions or []
        candidates = repository.sessions_for_selection(regions_selected, subjects_selected)
        updated_value = [session for session in current_sessions if session in candidates]
        return _options(candidates), updated_value

    @app.callback(
        Output("y-axis-range", "disabled"),
        Input("y-axis-mode", "value"),
    )
    def toggle_yaxis_range(mode):
        return mode != "manual"

    @app.callback(
        Output("timecourse-graph", "figure"),
        Output("counts-container", "children"),
        Output("last-loaded-label", "children"),
        Input("target-filter", "value"),
        Input("aggregator-radio", "value"),
        Input("region-filter", "value"),
        Input("subject-filter", "value"),
        Input("session-filter", "value"),
        Input("subset-filter", "value"),
        Input("block-type-filter", "value"),
        Input("y-axis-mode", "value"),
        Input("y-axis-range", "value"),
        Input("reload-data", "n_clicks"),
    )
    def update_figure(
        target_value,
        aggregator_choice,
        regions_selected,
        subjects_selected,
        sessions_selected,
        subsets_selected,
        block_types_selected,
        y_axis_mode,
        y_axis_range,
        _reload_clicks,
    ):
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        if triggered == "reload-data":
            repository.reload()

        dataset = repository.get_dataset()
        group_field = AGGREGATION_FIELDS.get(aggregator_choice, "region")
        regions_selected = regions_selected or []
        subjects_selected = subjects_selected or []
        sessions_selected = sessions_selected or []
        subsets_selected = subsets_selected or []
        block_types_selected = block_types_selected or [0, 1]

        timecourse_df = aggregate_timecourse(
            dataset.timecourse,
            group_field=group_field,
            target=target_value,
            regions=regions_selected,
            subjects=subjects_selected,
            sessions=sessions_selected,
            subsets=subsets_selected if group_field != "cell_subset" else subsets_selected,
            block_types=[int(bt) for bt in block_types_selected],
        )
        figure = _build_timecourse_figure(
            timecourse_df,
            group_field=group_field,
            y_mode=y_axis_mode,
            y_range=y_axis_range or [],
            summary_info=compute_selection_summary(
                dataset.metrics,
                target=target_value,
                regions=regions_selected,
                subjects=subjects_selected,
                sessions=sessions_selected,
                subsets=subsets_selected,
            ),
        )

        counts_df = aggregate_counts(
            dataset.metrics,
            group_field=group_field,
            target=target_value,
            regions=regions_selected,
            subjects=subjects_selected,
            sessions=sessions_selected,
            subsets=subsets_selected,
        )
        count_records = _prepare_count_records(counts_df, group_field)
        counts_table = _build_counts_table(count_records, group_field)

        return figure, counts_table, _format_timestamp(repository.last_loaded)

    return app
