from __future__ import annotations

import plotly.graph_objects as go

from traceshap.models.outcome import StepAttribution


def force_plot(
    attributions: list[StepAttribution],
    base_value: float = 0.5,
    title: str = "TraceSHAP Force Plot",
) -> go.Figure:
    sorted_attrs = sorted(attributions, key=lambda a: abs(a.quality_delta or 0), reverse=True)

    names = []
    values = []
    colors = []
    for attr in sorted_attrs:
        delta = attr.quality_delta or 0.0
        names.append(attr.step_name)
        values.append(delta)
        colors.append("rgba(255, 77, 77, 0.8)" if delta >= 0 else "rgba(77, 148, 255, 0.8)")

    cumulative = base_value
    x_starts = []
    x_ends = []
    for v in values:
        x_starts.append(cumulative)
        cumulative += v
        x_ends.append(cumulative)

    fig = go.Figure()

    for i in range(len(names)):
        fig.add_trace(go.Bar(
            y=[names[i]],
            x=[values[i]],
            base=[x_starts[i]],
            orientation="h",
            marker_color=colors[i],
            name=names[i],
            hovertemplate=(
                f"<b>{names[i]}</b><br>"
                f"quality_delta: {values[i]:.4f}<br>"
                f"<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.add_vline(x=base_value, line_dash="dash", line_color="gray",
                  annotation_text=f"base: {base_value:.3f}")
    fig.add_vline(x=cumulative, line_dash="solid", line_color="black",
                  annotation_text=f"final: {cumulative:.3f}")

    fig.update_layout(
        title=title,
        xaxis_title="Quality Score",
        yaxis_title="Step",
        barmode="overlay",
        height=max(300, len(names) * 50),
        template="plotly_white",
    )

    return fig
