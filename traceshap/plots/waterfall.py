from __future__ import annotations

import plotly.graph_objects as go

from traceshap.models.outcome import StepAttribution


def waterfall_plot(
    attributions: list[StepAttribution],
    base_value: float = 0.5,
    title: str = "TraceSHAP Waterfall Plot",
) -> go.Figure:
    names = ["Base Value"]
    measures = ["absolute"]
    values = [base_value]
    text = [f"{base_value:.4f}"]

    for attr in attributions:
        delta = attr.quality_delta or 0.0
        names.append(attr.step_name)
        measures.append("relative")
        values.append(delta)
        text.append(f"{delta:+.4f}")

    final_value = base_value + sum(a.quality_delta or 0.0 for a in attributions)
    names.append("Final Value")
    measures.append("total")
    values.append(final_value)
    text.append(f"{final_value:.4f}")

    fig = go.Figure(go.Waterfall(
        name="Attribution",
        orientation="v",
        measure=measures,
        x=names,
        y=values,
        text=text,
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)", "width": 1}},
        increasing={"marker": {"color": "rgba(255, 77, 77, 0.8)"}},
        decreasing={"marker": {"color": "rgba(77, 148, 255, 0.8)"}},
        totals={"marker": {"color": "rgba(100, 100, 100, 0.6)"}},
    ))

    fig.update_layout(
        title=title,
        yaxis_title="Quality Score",
        xaxis_title="Step",
        template="plotly_white",
        height=500,
    )

    return fig
