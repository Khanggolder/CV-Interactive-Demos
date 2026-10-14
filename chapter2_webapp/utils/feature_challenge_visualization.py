"""Plotly views for actual Feature Matching Challenge arrays and decisions."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from utils.feature_challenge import Pair, Result


def pair_view(pair: Pair, result: Result | None = None, *, filter_mode: str = "Tất cả ứng viên",
              selected: int | None = None, reveal_rule: bool = False,
              reveal_geometry: bool = False, presentation: bool = False) -> go.Figure:
    h, w = pair.a.shape
    canvas = np.concatenate((pair.a, np.ones((h, 12))*.95, pair.b), axis=1)
    gray = np.rint(canvas*255).astype(np.uint8)
    fig = go.Figure(go.Image(z=np.repeat(gray[:, :, None], 3, axis=2)))
    fig.update_traces(hoverinfo="skip")
    fig.add_annotation(x=w/2, y=-9, text="ẢNH A · nguồn", showarrow=False,
                       font=dict(size=17 if presentation else 14, color="#172033"))
    fig.add_annotation(x=w+12+w/2, y=-9, text="ẢNH B · biến đổi", showarrow=False,
                       font=dict(size=17 if presentation else 14, color="#172033"))
    if result is not None:
        decisions = list(enumerate(result.decisions))
        if filter_mode != "Tất cả ứng viên" and reveal_rule:
            decisions = [(i, d) for i, d in decisions if d.accepted]
        # Limit visual clutter only; metrics are computed from the entire set.
        decisions = decisions[:24]
        if (selected is not None and all(i != selected for i, _ in decisions)
                and (filter_mode == "Tất cả ứng viên" or result.decisions[selected].accepted)):
            decisions = decisions[:23] + [(selected, result.decisions[selected])]
        for i, d in decisions:
            if d.train is None:
                continue
            source = result.a.features[d.query]
            target = result.b.features[d.train]
            color = "#61758d"
            if reveal_rule and not d.accepted:
                color = "#9ca3af"
            if reveal_geometry and d.accepted:
                color = "#168653" if d.correct else "#d35425"
            width = 3.5 if i == selected else (1.6 if presentation else 1.25)
            opacity = .95 if i == selected else .55
            fig.add_shape(type="line", x0=source.x, y0=source.y,
                          x1=target.x+w+12, y1=target.y,
                          line=dict(color=color, width=width), opacity=opacity)
            if i == selected:
                fig.add_trace(go.Scatter(x=[source.x, target.x+w+12], y=[source.y, target.y],
                                         mode="markers+text", text=["A", "B"], textposition="top center",
                                         marker=dict(size=16 if presentation else 12, color=color,
                                                     line=dict(color="white", width=2)),
                                         textfont=dict(size=15, color="#172033"), showlegend=False,
                                         hovertemplate="(%{x:.1f}, %{y:.1f})<extra></extra>"))
    fig.update_layout(height=480 if presentation else 415, margin=dict(l=8, r=8, t=28, b=8),
                      paper_bgcolor="white", plot_bgcolor="white", dragmode=False)
    fig.update_xaxes(range=[-4, 2*w+17], visible=False, constrain="domain")
    fig.update_yaxes(range=[h+3, -17], visible=False, scaleanchor="x", scaleratio=1)
    return fig


def mask_view(pair: Pair) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=pair.valid_b.astype(int), colorscale=[[0,"#d35425"],[.5,"#d35425"],[.5,"#168653"],[1,"#168653"]],
                               showscale=False, hovertemplate="x=%{x}, y=%{y}: %{z}<extra></extra>"))
    fig.update_layout(height=330, margin=dict(l=25,r=8,t=12,b=20), paper_bgcolor="white")
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


def feature_view(result: Result, side: str = "A", presentation: bool = False) -> go.Figure:
    image = result.pair.a if side == "A" else result.pair.b
    detection = result.a if side == "A" else result.b
    gray = np.rint(image*255).astype(np.uint8)
    fig = go.Figure(go.Image(z=np.repeat(gray[:, :, None], 3, axis=2)))
    valid = [p for p in detection.features if p.descriptor is not None]
    invalid = [p for p in detection.features if p.descriptor is None]
    for points, color, label, symbol in ((invalid,"#a6aab1","descriptor không hợp lệ","x"),
                                          (valid,"#e85d04","descriptor hợp lệ","circle-open")):
        if points:
            fig.add_trace(go.Scatter(x=[p.x for p in points], y=[p.y for p in points],
                                     mode="markers", name=label,
                                     marker=dict(size=12 if presentation else 9, color=color,
                                                 symbol=symbol, line=dict(width=2))))
    fig.update_layout(height=425 if presentation else 330, margin=dict(l=8,r=8,t=8,b=8),
                      legend=dict(orientation="h",y=-.08), paper_bgcolor="white")
    fig.update_yaxes(range=[image.shape[0],-1], scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_xaxes(range=[-1,image.shape[1]], constrain="domain")
    return fig
