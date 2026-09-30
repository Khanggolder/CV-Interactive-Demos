"""Plotly helpers shared by the teaching demos."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def histogram_figure(
    image: np.ndarray,
    mode: str = "Grayscale",
    title: str = "Histogram",
    line_color: str = "#334155",
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
            go.Scatter(
                x=np.arange(256),
                y=counts,
                mode="lines",
                line={"color": line_color, "width": 2.5},
            )
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
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        font={"size": 15, "color": "#111827"},
        xaxis={
            "title": "Cường độ",
            "range": [0, 255],
            "fixedrange": True,
            "tickfont": {"size": 14},
            "gridcolor": "#dbe4ea",
            "linecolor": "#94a3b8",
        },
        yaxis={
            "title": "Số pixel",
            "fixedrange": True,
            "tickfont": {"size": 14},
            "gridcolor": "#dbe4ea",
            "linecolor": "#94a3b8",
        },
        showlegend=mode != "Grayscale" and image.ndim == 3,
        legend={"orientation": "h", "y": 1.15},
    )
    return figure


def gamma_curve_figure(
    gamma: float, input_intensity: int | None = None
) -> go.Figure:
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
    if input_intensity is not None:
        output_intensity = 255.0 * (input_intensity / 255.0) ** gamma
        figure.add_trace(
            go.Scatter(
                x=[input_intensity],
                y=[output_intensity],
                name=f"{input_intensity} → {output_intensity:.1f}",
                mode="markers",
                marker={
                    "size": 13,
                    "color": "#dc2626",
                    "line": {"color": "white", "width": 2},
                },
            )
        )
    figure.update_layout(
        title={"text": "Đường cong ánh xạ", "x": 0.02, "font": {"size": 19}},
        height=360,
        margin={"l": 50, "r": 20, "t": 48, "b": 45},
        template="plotly_white",
        xaxis={
            "title": "I_in",
            "range": [0, 255],
            "fixedrange": True,
            "gridcolor": "#dbe4ea",
            "linecolor": "#94a3b8",
        },
        yaxis={
            "title": "I_out",
            "range": [0, 255],
            "fixedrange": True,
            "gridcolor": "#dbe4ea",
            "linecolor": "#94a3b8",
        },
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        font={"size": 15, "color": "#111827"},
        legend={"orientation": "h", "y": 1.14},
    )
    return figure
