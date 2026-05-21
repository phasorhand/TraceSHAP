from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from traceshap.models.outcome import StepAttribution


def bar_plot(
    multi_trajectory_attrs: list[list[StepAttribution]],
    top_k: int = 10,
    title: str = "TraceSHAP Step Importance",
) -> go.Figure:
    step_values: dict[str, list[float]] = defaultdict(list)

    for traj_attrs in multi_trajectory_attrs:
        for attr in traj_attrs:
            step_values[attr.step_name].append(attr.quality_delta or 0.0)

    mean_abs = {name: sum(abs(v) for v in vals) / len(vals)
                for name, vals in step_values.items()}
    mean_signed = {name: sum(vals) / len(vals) for name, vals in step_values.items()}

    sorted_names = sorted(mean_abs, key=mean_abs.get, reverse=True)[:top_k]
    sorted_names.reverse()

    bar_values = [mean_abs[n] for n in sorted_names]
    bar_colors = [
        "rgba(255, 77, 77, 0.8)" if mean_signed[n] >= 0 else "rgba(77, 148, 255, 0.8)"
        for n in sorted_names
    ]

    fig = go.Figure(go.Bar(
        y=sorted_names,
        x=bar_values,
        orientation="h",
        marker_color=bar_colors,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "mean |SHAP|: %{x:.4f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Step Type",
        template="plotly_white",
        height=max(300, len(sorted_names) * 40),
    )

    return fig
