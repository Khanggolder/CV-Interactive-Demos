"""Plotly views of the pure Sobel state; no arithmetic is invented here."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .sobel import normalize_for_visualization, screen_vector


def matrix_figure(values, title: str, *, center=None, patch=False, active=None,
                  layer="Image", presentation=False, height=None) -> go.Figure:
    raw = np.asarray(values, dtype=float)
    if layer in ("Gx", "Gy"):
        shown = normalize_for_visualization(raw, signed=True)
        colors = [[0, "#e85d2a"], [0.5, "#f8fafc"], [1, "#14966f"]]
    elif layer == "Direction":
        # Cyclic HSV makes -180° and +180° share a hue.
        shown = (raw + 180) / 360
        colors = [[0, "#e85d2a"], [0.25, "#b392f0"], [0.5, "#2a9d8f"],
                  [0.75, "#b392f0"], [1, "#e85d2a"]]
    else:
        shown = raw / 255.0 if layer == "Image" else normalize_for_visualization(raw)
        colors = [[0, "#172b43"], [1, "#dbe9f0"]] if layer == "Image" else [[0, "#eef5f8"], [1, "#1996c8"]]
    h, w = raw.shape
    figure = go.Figure(go.Heatmap(z=shown, x=list(range(w)), y=list(range(h)),
                                  colorscale=colors, zmin=0, zmax=1, showscale=False,
                                  hovertemplate="row=%{y}, col=%{x}<extra></extra>"))
    for row in range(h):
        for col in range(w):
            val = raw[row, col]
            label = "—" if not np.isfinite(val) else f"{val:.0f}" if layer != "Direction" else f"{val:.0f}°"
            color = "white" if layer == "Image" and val < 125 else "#172033"
            if center == (row, col):
                color = "#ffd166" if layer == "Image" and val < 125 else (
                    "white" if layer != "Image" and np.isfinite(shown[row, col]) and shown[row, col] > 0.65
                    else "#172033")
            figure.add_annotation(x=col, y=row, text=label, showarrow=False,
                                  font=dict(size=19 if presentation else 14, color=color))
    if center is not None:
        row, col = center
        width = 3 if patch else 1
        figure.add_shape(type="rect", x0=col-width/2, x1=col+width/2,
                         y0=row-width/2, y1=row+width/2,
                         line=dict(color="#00b9d6", width=4 if presentation else 3))
    if active is not None and center is not None:
        row, col = center
        ar, ac = active
        figure.add_shape(type="rect", x0=col+ac-1.5, x1=col+ac-0.5,
                         y0=row+ar-1.5, y1=row+ar-0.5,
                         line=dict(color="#ffd166", width=4))
    figure.update_layout(title=dict(text=title, font=dict(size=23 if presentation else 18)),
                         height=height or (430 if presentation else 370), margin=dict(l=35,r=12,t=48,b=42),
                         paper_bgcolor="white", plot_bgcolor="white",
                         xaxis=dict(title="column →", range=[-0.5,w-0.5], tickmode="linear", dtick=1, side="bottom"),
                         yaxis=dict(title="row ↓", range=[h-0.5,-0.5], tickmode="linear", dtick=1),
                         annotations=figure.layout.annotations)
    figure.update_traces(hoverinfo="skip")
    return figure


def vector_figure(gx: int, gy: int, *, presentation=False) -> go.Figure:
    vx, vy = screen_vector(gx, gy)
    scale = max(abs(vx), abs(vy), 1)
    x, y = vx / scale, vy / scale
    fig = go.Figure()
    fig.add_hline(y=0, line_color="#94a3b8")
    fig.add_vline(x=0, line_color="#94a3b8")
    if vx or vy:
        fig.add_annotation(x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=4,
                           arrowcolor="#14966f")
    else:
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=14,color="#14966f"),
                                 showlegend=False))
    fig.update_layout(title="∇I · vector trên màn hình", height=360 if presentation else 290,
                      margin=dict(l=35,r=15,t=50,b=40), plot_bgcolor="#f8fafc", paper_bgcolor="white",
                      font=dict(size=17 if presentation else 13),
                      xaxis=dict(title="column → / Gx", range=[-1.3,1.3], zeroline=False),
                      yaxis=dict(title="screen ↑ = −Gy", range=[-1.3,1.3], scaleanchor="x", zeroline=False))
    return fig
