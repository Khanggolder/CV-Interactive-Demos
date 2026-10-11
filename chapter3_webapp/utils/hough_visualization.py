"""Synchronized Plotly views of the numerical Hough model."""

from __future__ import annotations

import numpy as np
import pandas as _pandas  # Plotly's optional validator is used by Streamlit.
import plotly.graph_objects as go

from .hough import Grid, line_segment, peak_geometry, signed_rho

BLUE = "#1996c8"
YELLOW = "#c48500"
ORANGE = "#e85d2a"
GREEN = "#14966f"
MUTED = "#94a3b8"


def _bounds(points) -> tuple[float, float, float, float]:
    array = np.asarray(points, dtype=float)
    return (min(-0.8, float(array[:, 0].min()) - 0.5),
            max(5.3, float(array[:, 0].max()) + 0.5),
            min(-0.8, float(array[:, 1].min()) - 0.5),
            max(4.0, float(array[:, 1].max()) + 0.5))


def _layout(figure: go.Figure, title: str, presentation: bool, height: int) -> None:
    figure.update_layout(
        title={"text": title, "x": 0.03, "font": {"size": 24 if presentation else 19}},
        height=height, template="plotly_white", paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font={"size": 18 if presentation else 13, "color": "#172033"},
        margin={"l": 55, "r": 16, "t": 60, "b": 70},
        legend={"orientation": "h", "y": -0.13, "x": 0, "yanchor": "top"},
        hovermode="closest",
    )


def image_figure(points, *, active: int = 0, theta: float | None = None,
                 visible: int | None = None, detected: tuple[float, float] | None = None,
                 ideal: bool = False, presentation: bool = False) -> go.Figure:
    """Candidate line, signed normal, points and bin-center detected line."""
    array = np.asarray(points, dtype=float)
    shown = array if visible is None else array[:visible]
    bounds = _bounds(array)
    figure = go.Figure()
    if ideal:
        ideal_segment = line_segment(np.sqrt(5), np.rad2deg(np.arctan2(2, 1)), bounds)
        if ideal_segment is not None:
            figure.add_trace(go.Scatter(x=ideal_segment[:, 0], y=ideal_segment[:, 1], mode="lines",
                                        name="Nguồn lý tưởng", line={"color": MUTED, "width": 3, "dash": "dash"}))
    if detected is not None:
        segment = line_segment(*detected, bounds)
        if segment is not None:
            figure.add_trace(go.Scatter(x=segment[:, 0], y=segment[:, 1], mode="lines",
                                        name="Peak → bin center", line={"color": GREEN, "width": 5}))
    if theta is not None and active < len(shown):
        rho = signed_rho(array[active], theta)
        segment = line_segment(rho, theta, bounds)
        if segment is not None:
            figure.add_trace(go.Scatter(x=segment[:, 0], y=segment[:, 1], mode="lines",
                                        name="Candidate qua điểm đang chọn", line={"color": YELLOW, "width": 4}))
        radians = np.deg2rad(theta)
        normal = np.array([np.cos(radians), np.sin(radians)])
        foot = rho * normal
        figure.add_trace(go.Scatter(x=[0, foot[0]], y=[0, foot[1]], mode="lines+markers",
                                    name="Signed ρ", line={"color": MUTED, "width": 3, "dash": "dot"},
                                    marker={"size": 6, "color": MUTED}))
        figure.add_annotation(x=float(0.85 * normal[0]), y=float(0.85 * normal[1]),
                              ax=0, ay=0, axref="x", ayref="y", text="",
                              showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor=ORANGE)
        figure.add_annotation(x=float(0.98 * normal[0]), y=float(0.98 * normal[1]),
                              text="n(θ)", showarrow=False, font={"color": ORANGE})
        figure.add_annotation(x=float(foot[0]), y=float(foot[1]), text=f"ρ={rho:.3f}",
                              showarrow=True, arrowhead=2, ax=25, ay=-32, font={"color": ORANGE})
    colors = [YELLOW if i == active else ORANGE if i == 1 else BLUE for i in range(len(shown))]
    sizes = [20 if presentation and i == active else 15 if i == active else 13 if presentation else 10
             for i in range(len(shown))]
    figure.add_trace(go.Scatter(x=shown[:, 0], y=shown[:, 1], mode="markers+text", name="Edge points",
                                text=[f"p{i+1}" if i < 2 or i == active else "" for i in range(len(shown))],
                                textposition="top center", marker={"size": sizes, "color": colors,
                                "line": {"width": 1.5, "color": "white"}},
                                customdata=np.arange(len(shown)),
                                hovertemplate="p%{customdata}<br>(%{x:.3f}, %{y:.3f})<extra></extra>"))
    _layout(figure, "IMAGE SPACE · y ↑", presentation, 650 if presentation else 470)
    figure.update_xaxes(title="x", range=bounds[:2], gridcolor="#dbe4ea", zerolinecolor=MUTED,
                        constrain="range")
    figure.update_yaxes(title="y", range=bounds[2:], gridcolor="#dbe4ea", zerolinecolor=MUTED,
                        scaleanchor="x", scaleratio=1)
    return figure


def continuous_figure(points, *, active: int = 0, theta: float = 0,
                      curves: int = 1, intersection: bool = False,
                      presentation: bool = False) -> go.Figure:
    array = np.asarray(points, dtype=float)
    figure = go.Figure()
    angles = np.linspace(0, 180, 721, endpoint=False)
    for index, point in enumerate(array[:curves]):
        color = YELLOW if index == active else ORANGE if index == 1 else BLUE
        figure.add_trace(go.Scatter(x=angles, y=signed_rho(point, angles), mode="lines",
                                    name=f"p{index+1} → ρ(θ)",
                                    showlegend=index < 2 or index == active,
                                    line={"color": color, "width": 4 if index == active else 2,
                                          "dash": "solid" if index == active else "dot"},
                                    hovertemplate="θ=%{x:.2f}°<br>ρ=%{y:.3f}<extra></extra>"))
    rho = signed_rho(array[active], theta)
    figure.add_trace(go.Scatter(x=[theta], y=[rho], mode="markers", name="(θ, ρ) hiện tại",
                                marker={"size": 18 if presentation else 14, "color": YELLOW,
                                        "line": {"color": "white", "width": 2}},
                                hovertemplate="θ=%{x:.2f}°<br>ρ=%{y:.3f}<extra></extra>"))
    if intersection:
        figure.add_trace(go.Scatter(x=[np.rad2deg(np.arctan2(2, 1))], y=[np.sqrt(5)],
                                    mode="markers", name="Giao điểm thực",
                                    marker={"size": 19 if presentation else 16, "color": GREEN,
                                            "symbol": "diamond"}))
    _layout(figure, "HOUGH SPACE · đường cong liên tục", presentation, 650 if presentation else 470)
    figure.update_xaxes(title="θ · normal (°)", range=[0, 180], gridcolor="#dbe4ea")
    extent = max(5.4, float(np.linalg.norm(array, axis=1).max(initial=0)) + 0.5)
    figure.update_yaxes(title="ρ · signed", range=[-extent, extent], gridcolor="#dbe4ea", zerolinecolor=MUTED)
    return figure


def accumulator_figure(counts: np.ndarray, grid: Grid, *, active_cell=None,
                       show_peak: bool = False, presentation: bool = False) -> go.Figure:
    rows, columns = grid.shape
    figure = go.Figure(go.Heatmap(
        z=counts, x=grid.theta_centers, y=grid.rho_centers, xgap=1, ygap=1,
        colorscale=[[0, "#edf3f6"], [0.2, "#a9dce5"], [0.6, "#47a2c0"], [1, GREEN]],
        zmin=0, zmax=max(6, int(counts.max(initial=0))),
        colorbar={"title": "Votes", "tickmode": "linear", "dtick": 1},
        customdata=np.array([[(row, col) for col in range(columns)] for row in range(rows)]),
        hovertemplate="A[%{customdata[0]},%{customdata[1]}]=%{z}"
                      "<br>θ center=%{x:.2f}°<br>ρ center=%{y:.2f}<extra></extra>",
    ))
    if active_cell is not None:
        row, col = active_cell
        figure.add_trace(go.Scatter(x=[grid.theta_centers[col]], y=[grid.rho_centers[row]],
                                    mode="markers", name="Vote vừa thêm", marker={"size": 18,
                                    "color": YELLOW, "symbol": "circle-open", "line": {"width": 3}}))
    if show_peak and counts.max(initial=0) > 0:
        from .hough import peaks
        _, winners = peaks(counts)
        figure.add_trace(go.Scatter(x=[grid.theta_centers[col] for _, col in winners],
                                    y=[grid.rho_centers[row] for row, _ in winners],
                                    mode="markers", name="Peak", marker={"size": 19,
                                    "color": GREEN, "symbol": "diamond-open", "line": {"width": 3}}))
    _layout(figure, "HOUGH SPACE · ACCUMULATOR", presentation, 650 if presentation else 470)
    figure.update_xaxes(title="θ bin center (°)", range=[0, 180], gridcolor="#dbe4ea")
    figure.update_yaxes(title="ρ bin center · signed", gridcolor="#dbe4ea")
    return figure
