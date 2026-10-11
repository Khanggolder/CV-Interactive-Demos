"""Plotly views of the computed Canny arrays; no numerical decisions here."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

STAGES = ("Original", "Smoothed", "Gx", "Gy", "Magnitude", "Gradient Direction",
          "NMS", "Double Threshold", "Final Edge Map")
COLORS = {
    "gray": [[0, "#0b1020"], [1, "#f8fafc"]],
    "gradient": [[0, "#0b1020"], [1, "#b68cff"]],
    "signed": [[0, "#47c2ff"], [.5, "#172033"], [1, "#ff8c42"]],
    "direction": [[0, "#b68cff"], [.25, "#47c2ff"], [.5, "#ffd166"], [.75, "#ff8c42"], [1, "#b68cff"]],
    "threshold": [[0, "#0b1020"], [.333, "#0b1020"], [.334, "#ff8c42"], [.666, "#ff8c42"], [.667, "#ffd166"], [1, "#ffd166"]],
    "replay": [[0, "#0b1020"], [.199, "#0b1020"], [.2, "#3d465c"], [.399, "#3d465c"],
               [.4, "#ff8c42"], [.599, "#ff8c42"], [.6, "#f8fafc"], [.799, "#f8fafc"],
               [.8, "#ffd166"], [1, "#ffd166"]],
}


def stage_data(arrays: dict, stage: str, frame: dict | None = None) -> tuple[np.ndarray, str, float, float]:
    if stage == "Original":
        return arrays["original"], "gray", 0, 1
    if stage == "Smoothed":
        return arrays["smoothed"], "gray", 0, 1
    if stage in ("Gx", "Gy"):
        data = arrays[stage.lower()]
        limit = max(float(np.max(np.abs(data))), 1e-9)
        return data, "signed", -limit, limit
    if stage == "Magnitude":
        return arrays["magnitude"], "gradient", 0, 1
    if stage == "Gradient Direction":
        data = np.rad2deg(arrays["direction"]) % 360
        data = np.where(arrays["magnitude_raw"] > 1e-12, data, np.nan)
        return data, "direction", 0, 360
    if stage == "NMS":
        return arrays["nms_normalized"], "gradient", 0, 1
    if stage == "Double Threshold":
        return arrays["weak"].astype(float) + 2 * arrays["strong"], "threshold", 0, 2
    if stage == "Final Edge Map":
        return arrays["edges"].astype(float), "gray", 0, 1
    if stage == "Hysteresis Replay" and frame is not None:
        values = np.zeros(arrays["strong"].shape)
        values[arrays["weak"] & ~arrays["edges"]] = 1
        values[arrays["weak"] & arrays["edges"]] = 2
        values[frame["accepted"] & arrays["weak"]] = 3
        values[arrays["strong"]] = 4
        return values, "replay", 0, 4
    raise ValueError(f"Unknown stage: {stage}")


def image_figure(arrays: dict, stage: str, *, point: tuple[int, int] | None = None,
                 comparison: dict | None = None, frame: dict | None = None,
                 presentation: bool = False, patch: bool = False) -> go.Figure:
    data, palette, zmin, zmax = stage_data(arrays, stage, frame)
    height, width = data.shape
    if patch:
        assert point is not None
        row, col = point
        radius = 3
        top, bottom = max(0, row - radius), min(height, row + radius + 1)
        left, right = max(0, col - radius), min(width, col + radius + 1)
        view = data[top:bottom, left:right]
        xs, ys = np.arange(left, right), np.arange(top, bottom)
    else:
        view, xs, ys = data, np.arange(width), np.arange(height)
    fig = go.Figure(go.Heatmap(z=view, x=xs, y=ys, colorscale=COLORS[palette], zmin=zmin, zmax=zmax,
                                showscale=False, hovertemplate="row=%{y}, col=%{x}<br>value=%{z:.4f}<extra></extra>",
                                zsmooth=False))
    if patch and max(view.shape) <= 7:
        fig.add_trace(go.Heatmap(z=np.zeros(view.shape), x=xs, y=ys, opacity=0, showscale=False,
                                  hoverinfo="skip", xgap=2, ygap=2))
    markers = []
    if comparison:
        markers.extend((comparison["q1"], "q1", "#47c2ff") for _ in (0,))
        markers.append((comparison["point"], "p", "#ffd166"))
        markers.append((comparison["q2"], "q2", "#47c2ff"))
    elif point is not None:
        markers.append((point, "p", "#ffd166"))
    if frame:
        if patch:
            for row, col in np.argwhere(frame["frontier"][ys[0]:ys[-1]+1, xs[0]:xs[-1]+1]):
                markers.append(((int(row + ys[0]), int(col + xs[0])), "queue", "#47c2ff"))
        if frame["active"] is not None:
            markers.append((frame["active"], "BFS", "#69d394"))
        for p in frame["added"][:8]:
            markers.append((p, "+weak", "#f8fafc"))
    for (row, col), label, color in markers:
        if not patch or (xs[0] <= col <= xs[-1] and ys[0] <= row <= ys[-1]):
            show_text = patch and label not in ("queue", "+weak")
            fig.add_trace(go.Scatter(x=[col], y=[row], mode="markers+text" if show_text else "markers",
                                     text=[label], textposition="top center", showlegend=False,
                                     textfont=dict(color="#f8fafc", size=14 if presentation else 11),
                                     marker=dict(symbol="square-open", size=31 if patch else (16 if presentation else 11),
                                                 color=color, line=dict(width=3, color=color)),
                                     hovertemplate=f"{label}: row={row}, col={col}<extra></extra>"))
    fig.update_layout(margin=dict(l=45, r=12, t=14, b=35), height=(490 if patch and presentation else
                      345 if patch else 760 if presentation else 420), paper_bgcolor="white",
                      plot_bgcolor="#0b1020", font=dict(size=17 if presentation else 13, color="#172033"),
                      showlegend=False)
    fig.update_xaxes(title="column x →", range=[xs[0] - .5, xs[-1] + .5],
                     dtick=1 if patch else None, constrain="domain")
    fig.update_yaxes(title="row y ↓", range=[ys[-1] + .5, ys[0] - .5],
                     dtick=1 if patch else None, scaleanchor="x", scaleratio=1,
                     constrain="domain")
    return fig


def kernel_figure(kernel: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=kernel, colorscale=COLORS["gradient"], showscale=False,
                                x=list(range(kernel.shape[1])), y=list(range(kernel.shape[0])),
                                text=np.round(kernel, 4), texttemplate="%{text:.4f}",
                                textfont=dict(size=17), xgap=2, ygap=2))
    fig.update_layout(height=340, margin=dict(l=30, r=10, t=10, b=30))
    fig.update_xaxes(range=[-.5, kernel.shape[1] - .5], dtick=1)
    fig.update_yaxes(range=[kernel.shape[0] - .5, -.5], dtick=1, scaleanchor="x")
    return fig
