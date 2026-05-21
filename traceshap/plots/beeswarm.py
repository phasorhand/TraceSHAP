from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from traceshap.models.outcome import StepAttribution


def beeswarm_plot(
    multi_trajectory_attrs: list[list[StepAttribution]],
    color_by: str = "cost_delta",
    title: str = "TraceSHAP Beeswarm Plot",
) -> go.Figure:
    step_values: dict[str, list[float]] = defaultdict(list)
    step_colors: dict[str, list[float]] = defaultdict(list)

    for traj_attrs in multi_trajectory_attrs:
        for attr in traj_attrs:
            step_values[attr.step_name].append(attr.quality_delta or 0.0)
            color_val = getattr(attr, color_by, None) or 0.0
            step_colors[attr.step_name].append(color_val)

    mean_abs = {name: sum(abs(v) for v in vals) / len(vals)
                for name, vals in step_values.items()}
    sorted_names = sorted(mean_abs, key=mean_abs.get, reverse=True)

    fig = go.Figure()

    for i, name in enumerate(sorted_names):
        vals = step_values[name]
        cols = step_colors[name]
        jitter = [i + (j % 5 - 2) * 0.08 for j in range(len(vals))]

        fig.add_trace(go.Scatter(
            x=vals,
            y=jitter,
            mode="markers",
            marker=dict(
                size=6,
                color=cols,
                colorscale="RdBu_r",
                showscale=(i == 0),
                colorbar=dict(title=color_by) if i == 0 else None,
                opacity=0.7,
            ),
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"quality_delta: %{{x:.4f}}<br>"
                f"{color_by}: %{{marker.color:.4f}}<br>"
                f"<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="SHAP Value (quality_delta)",
        yaxis=dict(
            tickvals=list(range(len(sorted_names))),
            ticktext=sorted_names,
            title="Step Type",
        ),
        template="plotly_white",
        height=max(400, len(sorted_names) * 60),
    )

    return fig
