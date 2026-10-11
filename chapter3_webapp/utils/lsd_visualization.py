"""Plotly views of measured LSD teaching-engine state; no detector decisions here."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

PURPLE = "#7543b5"
CYAN = "#087fa8"
YELLOW = "#efb000"
GREEN = "#14966f"
ORANGE = "#d45b20"
GRAY = "#64748b"


def field_figure(data, *, view="Intensity", selected=None, region=None, replay=None,
                 fit=False, rectangle=False, all_segments=False, nfa=False, show_decision=False,
                 presentation=False):
    intensity = data["intensity"]
    height, width = intensity.shape
    if view == "Magnitude":
        values = data["magnitude"]
        colors = [[0, "#f8f4ff"], [.4, "#bd9ae2"], [1, "#4a247d"]]
    else:
        span = float(np.ptp(intensity))
        values = (intensity-intensity.min()) / (span if span else 1)
        colors = [[0, "#15233a"], [1, "#dce6ec"]]
    fig = go.Figure(go.Heatmap(z=values, x=np.arange(width), y=np.arange(height),
                               colorscale=colors, showscale=False, zmin=0,
                               zmax=float(np.max(values)) if view == "Magnitude" and np.max(values) > 0 else 1,
                               customdata=intensity, hovertemplate="(y,x)=(%{y},%{x})<br>I=%{customdata:.4f}<extra></extra>"))
    # Shape overlays keep the pixel support distinguishable from normalized display colors.
    if region is not None:
        if nfa:
            aligned = region["fit"]["aligned"]
            pairs = zip(region["pixels"], aligned)
        elif replay is not None:
            pairs = ((p, True) for p in replay["members"])
        else:
            pairs = ((p, True) for p in region["pixels"])
        for (y, x), ok in pairs:
            color = CYAN if ok else ORANGE
            fig.add_shape(type="rect", x0=x-.47, x1=x+.47, y0=y-.47, y1=y+.47,
                          line_color=color, line_width=3.4 if presentation else 2.6,
                          fillcolor=color, opacity=.28)
        if replay is not None and replay["current"] is not None:
            event = replay["current"]
            y, x = event["pixel"]
            fig.add_shape(type="rect", x0=x-.5, x1=x+.5, y0=y-.5, y1=y+.5,
                          line_color=YELLOW if event["accepted"] else ORANGE, line_width=4)
            py, px = event["parent"]
            fig.add_trace(go.Scatter(x=[px, x], y=[py, y], mode="lines+markers",
                                     line=dict(color=YELLOW, width=3, dash="dot"),
                                     marker=dict(size=7), showlegend=False,
                                     hovertemplate="BFS parent → neighbor<extra></extra>"))
        sy, sx = region["seed"]
        fig.add_trace(go.Scatter(x=[sx], y=[sy], mode="markers", showlegend=False,
                                 marker=dict(symbol="star", size=20 if presentation else 16,
                                             color=YELLOW, line=dict(color="#493607", width=1)),
                                 hovertemplate="Seed (y,x)=(%{y},%{x})<extra></extra>"))
        if fit:
            fitted = region["fit"]
            endpoints = fitted["endpoints"]
            line_color = (GREEN if region.get("accept") else ORANGE) if show_decision else YELLOW
            fig.add_trace(go.Scatter(x=endpoints[:, 0], y=endpoints[:, 1], mode="lines+markers",
                                     line=dict(color=line_color,
                                               width=6 if presentation else 4),
                                     marker=dict(size=7), name="Finite segment"))
            center = fitted["center"]
            fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], mode="markers",
                                     marker=dict(color="#ec3a61", size=12, symbol="cross"),
                                     name="Weighted centroid"))
            if rectangle:
                corners = np.vstack((fitted["rectangle"], fitted["rectangle"][0]))
                fig.add_trace(go.Scatter(x=corners[:, 0], y=corners[:, 1], mode="lines",
                                         line=dict(color=GRAY, width=3, dash="dash"),
                                         name="Support rectangle"))
    if all_segments:
        for candidate in data["regions"]:
            if candidate["accept"]:
                endpoints = candidate["fit"]["endpoints"]
                fig.add_trace(go.Scatter(x=endpoints[:, 0], y=endpoints[:, 1], mode="lines",
                                         line=dict(color=GREEN, width=5 if presentation else 3),
                                         hovertemplate=f"n={candidate['fit']['n']}, k={candidate['fit']['k']}, NFA={candidate['nfa']:.3g}<extra></extra>",
                                         showlegend=False))
    if view in ("Gradient arrows", "Level lines"):
        xs, ys = [], []
        magnitude = data["magnitude"]
        for y in range(height):
            for x in range(width):
                if magnitude[y, x] < data["min_magnitude"] or magnitude[y, x] <= 1e-12:
                    continue
                if view == "Gradient arrows":
                    direction = np.array((data["gx"][y, x], data["gy"][y, x])) / magnitude[y, x]
                    fig.add_annotation(x=x+.31*direction[0], y=y+.31*direction[1],
                                       ax=x-.31*direction[0], ay=y-.31*direction[1],
                                       xref="x", yref="y", axref="x", ayref="y", text="",
                                       showarrow=True, arrowhead=2, arrowsize=.8,
                                       arrowwidth=2.5 if presentation else 1.8, arrowcolor=PURPLE)
                else:
                    angle = data["angles"][y, x]
                    dx, dy = .32*np.cos(angle), .32*np.sin(angle)
                    xs.extend((x-dx, x+dx, None))
                    ys.extend((y-dy, y+dy, None))
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False,
                                     line=dict(color=CYAN, width=3 if presentation else 2),
                                     hoverinfo="skip"))
    if selected is not None:
        y, x = selected
        fig.add_shape(type="rect", x0=x-.5, x1=x+.5, y0=y-.5, y1=y+.5,
                      line_color=YELLOW, line_width=4, fillcolor="rgba(0,0,0,0)")
    fig.update_layout(height=(min(760, max(510, 49*height)) if presentation else min(640, max(490, 45*height))),
                      margin=dict(l=28, r=18, t=16, b=28), paper_bgcolor="white",
                      plot_bgcolor="white", font=dict(size=17 if presentation else 13),
                      legend=dict(orientation="h", y=-.12))
    x_bounds = [-.5, width-.5]
    y_bounds = [-.5, height-.5]
    if region is not None and fit and rectangle:
        corners = np.asarray(region["fit"]["rectangle"])
        x_bounds = [min(x_bounds[0], float(corners[:, 0].min())-.25),
                    max(x_bounds[1], float(corners[:, 0].max())+.25)]
        y_bounds = [min(y_bounds[0], float(corners[:, 1].min())-.25),
                    max(y_bounds[1], float(corners[:, 1].max())+.25)]
    fig.update_xaxes(range=x_bounds, dtick=1, title="x · cột",
                     showgrid=True, gridcolor="rgba(90,115,140,.18)", constrain="domain")
    fig.update_yaxes(range=y_bounds, dtick=1, title="y · hàng (tăng lên)",
                     showgrid=True, gridcolor="rgba(90,115,140,.18)",
                     scaleanchor="x", scaleratio=1, constrain="domain")
    return fig
