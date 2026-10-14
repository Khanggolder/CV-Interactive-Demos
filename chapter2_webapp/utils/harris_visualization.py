"""Plotly figures for computed Chapter 2 data; no Streamlit state here."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def heatmap(values: np.ndarray, title: str, *, center: tuple[int, int] | None = None,
            detections: np.ndarray | None = None, presentation: bool = False,
            colorscale: str = "Viridis", displacement: bool = False) -> go.Figure:
    values = np.asarray(values)
    h, w = values.shape
    xs = np.arange(w) - w//2 if displacement else np.arange(w)
    ys = np.arange(h) - h//2 if displacement else np.arange(h)
    fig = go.Figure(go.Heatmap(z=values, x=xs, y=ys,
                               colorscale=colorscale, colorbar=dict(len=0.78, thickness=12),
                               hovertemplate="x=%{x}, y=%{y}<br>Giá trị=%{z:.5g}<extra></extra>"))
    if detections is not None:
        rows, cols = np.nonzero(detections)
        fig.add_trace(go.Scatter(x=cols, y=rows, mode="markers", name="Góc giữ lại",
                                 marker=dict(symbol="circle-open", color="#e85d2a",
                                             size=17 if presentation else 12, line=dict(width=3))))
    if center is not None:
        fig.add_trace(go.Scatter(x=[center[1]], y=[center[0]], mode="markers", name="Pixel đang xét",
                                 marker=dict(symbol="cross", color="#f43f5e",
                                             size=19 if presentation else 14, line=dict(width=3))))
    fig.update_layout(title=dict(text=title, font=dict(size=22 if presentation else 17)),
                      height=550 if presentation else 390,
                      margin=dict(l=34, r=25, t=55, b=66 if detections is not None else 42),
                      font=dict(size=16 if presentation else 12, color="#172033"),
                      paper_bgcolor="white", plot_bgcolor="white", showlegend=detections is not None,
                      legend=dict(orientation="h", x=0, y=-0.20, xanchor="left", yanchor="top",
                                  bgcolor="rgba(255,255,255,.9)"))
    fig.update_xaxes(title="u · dịch x" if displacement else "x = cột",
                     range=[xs[0]-.5, xs[-1]+.5], constrain="domain")
    fig.update_yaxes(title="v · dịch y (xuống)" if displacement else "y = hàng (hướng xuống)",
                     range=[ys[-1]+.5, ys[0]-.5], scaleanchor="x",
                     scaleratio=1)
    return fig


def eigenview(large: float, small: float, presentation: bool = False) -> go.Figure:
    """Computed eigenvalue-space point; no finite ellipse is drawn for zero eigenvalues."""
    scale = max(large, 0.005) * 1.2
    fig = go.Figure(go.Scatter(x=[small], y=[large], mode="markers+text",
                               text=[f"(λ₂={small:.4g}, λ₁={large:.4g})"],
                               textposition="top center", marker=dict(size=20, color="#e85d2a")))
    fig.update_layout(title="Không gian eigenvalue", height=470 if presentation else 340,
                      xaxis=dict(title="λ₂ (nhỏ)", range=[0, scale]),
                      yaxis=dict(title="λ₁ (lớn)", range=[0, scale]),
                      margin=dict(l=40, r=30, t=55, b=45),
                      font=dict(size=17 if presentation else 12), paper_bgcolor="white")
    return fig
