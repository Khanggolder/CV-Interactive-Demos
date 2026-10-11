"""Synchronized Cartesian Plotly views for circle geometry and real votes."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .hough_circle import Dataset, Grid, VoteState, candidate_center, center_locus, cell_parameters, peaks

BLUE = "#1996c8"
YELLOW = "#bc7900"
ORANGE = "#e85d2a"
PURPLE = "#8057b3"
GREEN = "#14966f"
GRAY = "#8191a5"
POINT_COLORS = (YELLOW, ORANGE, PURPLE)


def _base(title: str, presentation: bool):
    fig = go.Figure()
    fig.update_layout(title={"text": title, "x": .03, "font": {"size": 24 if presentation else 19}},
                      height=650 if presentation else 475, template="plotly_white",
                      paper_bgcolor="white", plot_bgcolor="#f8fafc",
                      font={"size": 18 if presentation else 13, "color": "#172033"},
                      margin={"l": 52, "r": 18, "t": 58, "b": 62},
                      legend={"orientation": "h", "y": -0.17, "x": 0, "yanchor": "top"},
                      hovermode="closest")
    return fig


def _trace_circle(fig, center, radius, *, color, name, dash="solid", width=2.5,
                  opacity=1, showlegend=True):
    phi = np.linspace(0, 360, 181)
    coords = candidate_center(center, radius, phi)
    fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="lines", name=name,
                             showlegend=showlegend, opacity=opacity,
                             line={"color": color, "width": width, "dash": dash},
                             hovertemplate=f"{name}<br>(%{{x:.3f}}, %{{y:.3f}})<extra></extra>"))


def _axes(fig, bounds, labels):
    xmin, xmax, ymin, ymax = bounds
    # Both data ranges and scaleanchor are explicit: no geometric ellipse.
    fig.update_xaxes(title=labels[0], range=[xmin, xmax], gridcolor="#dbe4ea",
                     zerolinecolor=GRAY, constrain="range")
    fig.update_yaxes(title=labels[1], range=[ymin, ymax], gridcolor="#dbe4ea",
                     zerolinecolor=GRAY, scaleanchor="x", scaleratio=1,
                     constrain="domain")


def _bounds(dataset: Dataset, radius: float, candidate, grid: Grid | None = None):
    points = np.array([p.xy for p in dataset.points])
    centers = np.array([circle[:2] for circle in dataset.true_circles]) if dataset.true_circles else points
    locations = np.vstack((points, centers, np.asarray(candidate)[None, :]))
    reach = max(radius, *(circle[2] for circle in dataset.true_circles)) + .7
    return (float(locations[:, 0].min()-reach), float(locations[:, 0].max()+reach),
            float(locations[:, 1].min()-reach), float(locations[:, 1].max()+reach))


def image_figure(dataset: Dataset, *, active=0, visible=None, radius=2., phi=-90.,
                 candidate=True, detected=None, show_truth=False, gradient=False,
                 gradient_polarity="inward", presentation=False):
    obs = dataset.points[active]
    center = candidate_center(obs.xy, radius, phi)
    fig = _base("IMAGE SPACE · (x, y), y ↑", presentation)
    # The reference outline is visible for observation; numerical truth labels wait for reveal.
    for i, truth in enumerate(dataset.true_circles):
        _trace_circle(fig, truth[:2], truth[2], color=BLUE if not show_truth else GRAY,
                      name="Vòng tròn nguồn" if not show_truth else "Nguồn lý tưởng",
                      dash="solid" if not show_truth else "dash", width=3,
                      opacity=.7 if not show_truth else .55, showlegend=i == 0)
        if show_truth:
            fig.add_trace(go.Scatter(x=[truth[0]], y=[truth[1]], mode="markers",
                                     name="Tâm nguồn", showlegend=i == 0,
                                     marker={"color": GRAY, "size": 9, "symbol": "x"}))
    if candidate:
        _trace_circle(fig, center, radius, color=YELLOW, name="Candidate circle", width=3.5)
        fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], mode="markers",
                                 name="Candidate center", marker={"color": YELLOW,
                                 "size": 17 if presentation else 13, "symbol": "diamond"}))
        fig.add_trace(go.Scatter(x=[obs.xy[0], center[0]], y=[obs.xy[1], center[1]],
                                 mode="lines", name="Bán kính r", showlegend=False,
                                 line={"color": GRAY, "width": 2, "dash": "dot"}))
    if detected is not None:
        _trace_circle(fig, detected[:2], detected[2], color=GREEN,
                      name="Detected · bin center", width=4)
        fig.add_trace(go.Scatter(x=[detected[0]], y=[detected[1]], mode="markers",
                                 name="Detected center", marker={"color": GREEN,
                                 "size": 18 if presentation else 14, "symbol": "diamond"}))
    shown = dataset.points if visible is None else dataset.points[:visible]
    fig.add_trace(go.Scatter(x=[p.xy[0] for p in shown], y=[p.xy[1] for p in shown],
                             text=[p.id if i < 3 or i == active else "" for i, p in enumerate(shown)],
                             mode="markers+text", textposition="top center", name="Observed points",
                             marker={"color": [YELLOW if p.id == obs.id else ORANGE if i == 1 else PURPLE if i == 2 else BLUE
                                               for i, p in enumerate(shown)],
                                     "size": [18 if presentation else 13 if p.id == obs.id else 11
                                              for p in shown], "line": {"color": "white", "width": 1.5}},
                             hovertemplate="(%{x:.4f}, %{y:.4f})<extra></extra>"))
    if gradient and obs.gradient is not None and np.linalg.norm(obs.gradient) > 1e-12:
        direction = np.asarray(obs.gradient) / np.linalg.norm(obs.gradient)
        end = np.asarray(obs.xy) + direction * min(1.25, radius * .65)
        fig.add_annotation(x=float(end[0]), y=float(end[1]), ax=float(obs.xy[0]), ay=float(obs.xy[1]),
                           axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=3,
                           arrowcolor=ORANGE)
    _axes(fig, _bounds(dataset, radius, center), ("x", "y"))
    return fig


def center_figure(dataset: Dataset, *, active=0, visible=1, radius=2., phi=-90.,
                  show_loci=True, intersections=(), grid: Grid | None = None,
                  votes: VoteState | None = None, layer=0, show_peak=False,
                  show_candidate=True, presentation=False):
    obs = dataset.points[active]
    center = candidate_center(obs.xy, radius, phi)
    fig = _base("CENTER SPACE · (a, b), y ↑", presentation)
    if grid is not None and votes is not None:
        counts = votes.counts[layer]
        fig.add_trace(go.Heatmap(z=counts, x=grid.a_centers, y=grid.b_centers,
                                 xgap=1, ygap=1, name="Support votes",
                                 colorscale=[[0, "#edf3f6"], [.2, "#a9dce5"], [.65, "#47a2c0"], [1, GREEN]],
                                 zmin=0, zmax=max(3, int(counts.max(initial=0))),
                                 colorbar={"title": "Vote", "thickness": 10, "len": .56, "xpad": 2},
                                 hovertemplate="a=%{x:.3f}, b=%{y:.3f}<br>support=%{z}<extra></extra>"))
    if show_loci:
        for i, point in enumerate(dataset.points[:visible]):
            locus = center_locus(point.xy, radius)
            fig.add_trace(go.Scatter(x=locus[:, 0], y=locus[:, 1], mode="lines",
                                     name=f"{point.id} → locus",
                                     showlegend=i < 3, opacity=1 if i < 3 else .38,
                                     line={"color": POINT_COLORS[i] if i < 3 else BLUE,
                                           "width": 3 if i == active else 2}))
    if show_candidate:
        fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], mode="markers",
                                 name="Candidate center", marker={"color": YELLOW,
                                 "size": 18 if presentation else 14, "symbol": "diamond"}))
    if len(intersections):
        coords = np.asarray(intersections)
        fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="markers",
                                 name="Giao loci liên tục", marker={"color": PURPLE,
                                 "size": 18 if presentation else 15, "symbol": "x",
                                 "line": {"width": 3}}))
    if votes is not None and show_peak and votes.counts.max(initial=0) > 0:
        maximum, cells = peaks(votes.counts)
        selected = [cell for cell in cells if cell[0] == layer]
        if selected:
            fig.add_trace(go.Scatter(x=[cell_parameters(grid, c)[0] for c in selected],
                                     y=[cell_parameters(grid, c)[1] for c in selected], mode="markers",
                                     name=f"Peak {maximum}", marker={"color": GREEN,
                                     "size": 21 if presentation else 17,
                                     "symbol": "diamond-open", "line": {"width": 3}}))
        if votes.last_event and votes.last_event.cell and votes.last_event.cell[0] == layer:
            _, row, col = votes.last_event.cell
            fig.add_trace(go.Scatter(x=[grid.a_centers[col]], y=[grid.b_centers[row]],
                                     mode="markers", name="Bin vừa thăm",
                                     marker={"color": ORANGE, "size": 18, "symbol": "square-open",
                                             "line": {"width": 3}}))
    bounds = (float(grid.a_edges[0]), float(grid.a_edges[-1]),
              float(grid.b_edges[0]), float(grid.b_edges[-1])) if grid is not None else _bounds(dataset, radius, center)
    _axes(fig, bounds, ("a · tâm x", "b · tâm y"))
    return fig
