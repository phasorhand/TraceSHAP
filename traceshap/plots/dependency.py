from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from traceshap.models.outcome import StepAttribution


def dependency_plot(
    multi_trajectory_attrs: list[list[StepAttribution]],
    step_name: str,
    color_by: str,
    title: str | None = None,
) -> go.Figure:
    if title is None:
        title = f"TraceSHAP Dependency: {step_name} vs {color_by}"

    x_values = []
    y_values = []
    color_values = []

    for traj_attrs in multi_trajectory_attrs:
        attr_map = {a.step_name: a for a in traj_attrs}
        target = attr_map.get(step_name)
        color_attr = attr_map.get(color_by)

        if target is None:
            continue

        x_val = target.cost_delta or 0.0
        y_val = target.quality_delta or 0.0
        c_val = (color_attr.quality_delta or 0.0) if color_attr else 0.0

        x_values.append(x_val)
        y_values.append(y_val)
        color_values.append(c_val)

    fig = go.Figure(go.Scatter(
        x=x_values,
        y=y_values,
        mode="markers",
        marker=dict(
            size=8,
            color=color_values,
            colorscale="RdBu_r",
            showscale=True,
            colorbar=dict(title=f"{color_by} SHAP"),
            opacity=0.8,
        ),
        hovertemplate=(
            f"<b>{step_name}</b><br>"
            f"cost_delta: %{{x:.4f}}<br>"
            f"quality_delta: %{{y:.4f}}<br>"
            f"{color_by} SHAP: %{{marker.color:.4f}}<br>"
            f"<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis_title=f"{step_name} cost_delta",
        yaxis_title=f"{step_name} SHAP Value",
        template="plotly_white",
        height=500,
    )

    return fig
