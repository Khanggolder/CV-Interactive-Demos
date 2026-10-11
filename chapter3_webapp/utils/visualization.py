"""Plotly figures for explaining computed RANSAC geometry."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as _pandas  # Ensure Plotly's optional validator dependency is fully initialized.
import plotly.graph_objects as go

from .ransac import RansacResult, Trial, normalize_line, perpendicular_feet


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
BLUE = "#1996c8"
ORANGE = "#e85d2a"
GOLD = "#f0b429"
GREEN = "#14966f"
VIOLET = "#7047b8"
NAVY = "#123b5d"
MUTED = "#94a3b8"
LAB_STAGES = ("Sample", "Fit", "Measure", "Consensus", "Update Best")


def _line_segment(
    model: np.ndarray, bounds: tuple[float, float, float, float]
) -> np.ndarray | None:
    """Clip an infinite implicit line to an axis-aligned rectangle."""
    line = normalize_line(model)
    xmin, xmax, ymin, ymax = bounds
    candidates: list[np.ndarray] = []
    a, b, c = line
    if abs(b) > 1e-12:
        for x_value in (xmin, xmax):
            y_value = -(a * x_value + c) / b
            if ymin - 1e-10 <= y_value <= ymax + 1e-10:
                candidates.append(np.array([x_value, y_value]))
    if abs(a) > 1e-12:
        for y_value in (ymin, ymax):
            x_value = -(b * y_value + c) / a
            if xmin - 1e-10 <= x_value <= xmax + 1e-10:
                candidates.append(np.array([x_value, y_value]))
    unique: list[np.ndarray] = []
    for point in candidates:
        if not any(np.allclose(point, existing) for existing in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    # The farthest pair handles corner intersections without ordering assumptions.
    pairs = [
        (np.linalg.norm(unique[i] - unique[j]), unique[i], unique[j])
        for i in range(len(unique))
        for j in range(i + 1, len(unique))
    ]
    _, first, second = max(pairs, key=lambda item: item[0])
    return np.vstack((first, second))


def _epsilon_polygon(
    model: np.ndarray,
    epsilon: float,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Clip ``|a*x+b*y+c| <= epsilon`` to the plotting rectangle."""
    line = normalize_line(model)
    xmin, xmax, ymin, ymax = bounds
    vertices = [
        np.array([xmin, ymin]),
        np.array([xmax, ymin]),
        np.array([xmax, ymax]),
        np.array([xmin, ymax]),
    ]
    for sign in (-1.0, 1.0):
        output: list[np.ndarray] = []
        if not vertices:
            break
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            first_value = sign * (first @ line[:2] + line[2]) - epsilon
            second_value = sign * (second @ line[:2] + line[2]) - epsilon
            if first_value <= 1e-12:
                output.append(first)
            if (first_value <= 0) != (second_value <= 0):
                fraction = first_value / (first_value - second_value)
                output.append(first + fraction * (second - first))
        vertices = output
    return np.asarray(vertices)


def _add_model(
    figure: go.Figure,
    model: np.ndarray,
    bounds: tuple[float, float, float, float],
    name: str,
    color: str,
    width: float = 3.5,
    dash: str = "solid",
) -> None:
    segment = _line_segment(model, bounds)
    if segment is not None:
        figure.add_trace(
            go.Scatter(
                x=segment[:, 0],
                y=segment[:, 1],
                mode="lines",
                name=name,
                line={"color": color, "width": width, "dash": dash},
                hoverinfo="name",
            )
        )


def point_cloud_figure(
    points: np.ndarray,
    *,
    title: str,
    model: np.ndarray | None = None,
    epsilon: float | None = None,
    inlier_mask: np.ndarray | None = None,
    sample_pair: tuple[int, int] | None = None,
    show_distances: bool = False,
    distance_indices: Sequence[int] | None = None,
    best_model: np.ndarray | None = None,
    refit_model: np.ndarray | None = None,
    ols_model: np.ndarray | None = None,
    presentation: bool = False,
    bounds: tuple[float, float, float, float] = (-4.5, 4.5, -3.4, 3.4),
) -> go.Figure:
    """Build a synchronized point/model/consensus teaching plot."""
    array = np.asarray(points, dtype=float)
    padding = 0.25
    bounds = (
        min(bounds[0], float(array[:, 0].min()) - padding),
        max(bounds[1], float(array[:, 0].max()) + padding),
        min(bounds[2], float(array[:, 1].min()) - padding),
        max(bounds[3], float(array[:, 1].max()) + padding),
    )
    figure = go.Figure()

    if model is not None and epsilon is not None:
        polygon = _epsilon_polygon(model, epsilon, bounds)
        if len(polygon) >= 3:
            closed = np.vstack((polygon, polygon[0]))
            figure.add_trace(
                go.Scatter(
                    x=closed[:, 0],
                    y=closed[:, 1],
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(20, 150, 111, 0.14)",
                    line={"color": "rgba(20, 150, 111, 0.35)", "width": 1},
                    name=f"Dải ε = {epsilon:.2f}",
                    hoverinfo="name",
                )
            )

    if show_distances and model is not None:
        feet = perpendicular_feet(array, model)
        selected = range(len(array)) if distance_indices is None else distance_indices
        for trace_index, index in enumerate(selected):
            point, foot = array[index], feet[index]
            figure.add_trace(
                go.Scatter(
                    x=[point[0], foot[0]],
                    y=[point[1], foot[1]],
                    mode="lines",
                    line={"color": "rgba(100,116,139,.55)", "width": 1, "dash": "dot"},
                    showlegend=trace_index == 0,
                    name="Khoảng cách vuông góc",
                    hoverinfo="skip",
                )
            )

    if inlier_mask is None:
        colors = np.full(len(array), BLUE, dtype=object)
        labels = np.full(len(array), "Chưa phân loại", dtype=object)
    else:
        mask = np.asarray(inlier_mask, dtype=bool)
        colors = np.where(mask, BLUE, ORANGE)
        labels = np.where(mask, "Inlier", "Outlier")

    sizes = np.full(len(array), 14.0 if presentation else 10.0)
    line_colors = np.full(len(array), "white", dtype=object)
    line_widths = np.full(len(array), 1.5)
    if sample_pair is not None:
        for index in sample_pair:
            sizes[index] = 21 if presentation else 16
            line_colors[index] = GOLD
            line_widths[index] = 4

    custom = np.column_stack((np.arange(len(array)), labels))
    figure.add_trace(
        go.Scatter(
            x=array[:, 0],
            y=array[:, 1],
            mode="markers",
            name="Điểm dữ liệu",
            customdata=custom,
            marker={
                "size": sizes,
                "color": colors,
                "line": {"color": line_colors, "width": line_widths},
            },
            hovertemplate=(
                "Điểm #%{customdata[0]}<br>x=%{x:.3f}<br>y=%{y:.3f}"
                "<br>%{customdata[1]}<extra></extra>"
            ),
        )
    )

    if model is not None:
        _add_model(figure, model, bounds, "Candidate hiện tại", GOLD, dash="dash" if best_model is not None else "solid")
    if best_model is not None:
        _add_model(figure, best_model, bounds, "Best consensus model", GREEN, 4.5)
    if refit_model is not None:
        _add_model(figure, refit_model, bounds, "Final TLS refit", VIOLET, 4.5)
    if ols_model is not None:
        _add_model(figure, ols_model, bounds, "OLS trên mọi điểm", ORANGE, 3, "dash")

    xmin, xmax, ymin, ymax = bounds
    figure.update_layout(
        title={"text": title, "x": 0.02, "font": {"size": 25 if presentation else 20}},
        height=720 if presentation else 560,
        margin={"l": 60 if presentation else 45, "r": 18, "t": 82 if presentation else 72, "b": 108 if presentation else 92},
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        font={"size": 19 if presentation else 15, "color": "#172033"},
        hovermode="closest",
        legend={
            "orientation": "h",
            "y": -0.16,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis={
            "title": "x",
            "range": [xmin, xmax],
            "gridcolor": "#dbe4ea",
            "zerolinecolor": "#94a3b8",
            "constrain": "domain",
        },
        yaxis={
            "title": "y",
            "range": [ymin, ymax],
            "gridcolor": "#dbe4ea",
            "zerolinecolor": "#94a3b8",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    )
    return figure


def lab_stage_figure(
    points: np.ndarray,
    result: RansacResult,
    epsilon: float,
    stage: str,
    *,
    presentation: bool = False,
) -> go.Figure:
    """Reveal one computed part of the current trial without changing RANSAC state."""
    if stage not in LAB_STAGES or not result.trials or result.best_trial is None:
        raise ValueError("a valid stage and at least one RANSAC trial are required")
    candidate = result.trials[-1]
    best = result.best_trial
    common = {"sample_pair": candidate.pair, "presentation": presentation}
    title = f"Iteration {candidate.iteration} · {stage}"
    if stage == "Sample":
        return point_cloud_figure(points, title=title, **common)
    if stage == "Fit":
        return point_cloud_figure(points, title=title, model=candidate.model, **common)
    if stage == "Measure":
        # At most two examples: nearest to ε on each side of the boundary.
        inside = np.flatnonzero(candidate.inlier_mask)
        outside = np.flatnonzero(~candidate.inlier_mask)
        examples = []
        for group in (inside, outside):
            non_sample = [int(index) for index in group if index not in candidate.pair]
            if non_sample:
                examples.append(min(non_sample, key=lambda index: abs(candidate.distances[index] - epsilon)))
        return point_cloud_figure(
            points, title=title, model=candidate.model, epsilon=epsilon,
            show_distances=True, distance_indices=examples, **common,
        )
    if stage == "Consensus":
        return point_cloud_figure(
            points, title=title, model=candidate.model, epsilon=epsilon,
            inlier_mask=candidate.inlier_mask, **common,
        )
    return point_cloud_figure(
        points, title=title, model=candidate.model, best_model=best.model,
        refit_model=result.refit_model, inlier_mask=best.inlier_mask, **common,
    )


def trial_score_figure(
    trials: Sequence[Trial], best_iteration: int | None, *, presentation: bool = False
) -> go.Figure:
    """Show the computed consensus score reached by each trial."""
    x_values = [trial.iteration for trial in trials]
    scores = [trial.score for trial in trials]
    colors = [GREEN if value == best_iteration else NAVY for value in x_values]
    figure = go.Figure(
        go.Bar(
            x=x_values,
            y=scores,
            marker_color=colors,
            text=scores,
            textposition="outside",
            hovertemplate="Iteration %{x}<br>Consensus %{y}<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": "Consensus qua từng lần lấy mẫu", "x": 0.02},
        height=360 if presentation else 290,
        margin={"l": 45, "r": 15, "t": 55, "b": 40},
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        font={"size": 18 if presentation else 14, "color": "#172033"},
        xaxis={"title": "Iteration", "dtick": 1, "gridcolor": "#edf2f7"},
        yaxis={"title": "Số inlier", "rangemode": "tozero", "gridcolor": "#dbe4ea"},
        showlegend=False,
    )
    return figure
