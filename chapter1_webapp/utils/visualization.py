"""Plotly helpers shared by the teaching demos."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def histogram_figure(
    image: np.ndarray, mode: str = "Grayscale", title: str = "Histogram"
) -> go.Figure:
    """Build a grayscale or RGB histogram with a projector-friendly style."""
    figure = go.Figure()
    if image.ndim == 2 or mode == "Grayscale":
        if image.ndim == 3:
            # Standard luminance coefficients, avoiding a cv2 dependency here.
            values = np.clip(
                0.299 * image[:, :, 0]
                + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2],
                0,
                255,
            ).astype(np.uint8)
        else:
            values = image
        counts = np.bincount(values.ravel(), minlength=256)
        figure.add_trace(
            go.Scatter(x=np.arange(256), y=counts, mode="lines", line={"color": "#334155", "width": 2})
        )
    else:
        for channel, color, label in zip(range(3), ("#ef4444", "#22c55e", "#3b82f6"), ("R", "G", "B")):
            counts = np.bincount(image[:, :, channel].ravel(), minlength=256)
            figure.add_trace(
                go.Scatter(
                    x=np.arange(256), y=counts, mode="lines", name=label,
                    line={"color": color, "width": 2}, opacity=0.82,
                )
            )
    figure.update_layout(
        title={"text": title, "x": 0.02, "font": {"size": 19}},
        height=275,
        margin={"l": 42, "r": 15, "t": 45, "b": 38},
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        xaxis={"title": "Cường độ", "range": [0, 255], "fixedrange": True},
        yaxis={"title": "Số pixel", "fixedrange": True},
        showlegend=mode != "Grayscale" and image.ndim == 3,
        legend={"orientation": "h", "y": 1.15},
    )
    return figure


def gamma_curve_figure(gamma: float) -> go.Figure:
    """Plot the current input-output intensity mapping."""
    x = np.arange(256, dtype=np.float32)
    y = 255.0 * np.power(x / 255.0, gamma)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=x, y=x, name="γ = 1", line={"color": "#94a3b8", "dash": "dash"})
    )
    figure.add_trace(
        go.Scatter(x=x, y=y, name=f"γ = {gamma:.2f}", line={"color": "#e85d04", "width": 4})
    )
    figure.update_layout(
        title={"text": "Đường cong ánh xạ", "x": 0.02, "font": {"size": 19}},
        height=360,
        margin={"l": 50, "r": 20, "t": 48, "b": 45},
        xaxis={"title": "I_in", "range": [0, 255], "fixedrange": True},
        yaxis={"title": "I_out", "range": [0, 255], "fixedrange": True},
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        legend={"orientation": "h", "y": 1.14},
    )
    return figure

