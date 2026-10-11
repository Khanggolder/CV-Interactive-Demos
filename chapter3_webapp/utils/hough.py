"""Pure Hough line geometry and deterministic, incremental accumulator voting.

Coordinates are Cartesian (y up). Theta is a normal angle in [0, 180),
rho is signed, and all accumulator intervals are half-open.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


P1 = (1.0, 2.0)
P2 = (3.0, 1.0)
ALIGNED = (P1, P2, (0.2, 2.4), (1.8, 1.6), (3.7, 0.65), (4.5, 0.25))
DISTRACTORS = ((0.5, 0.3), (2.5, 3.3), (4.1, 2.6))
DEFAULT_POINTS = ALIGNED + DISTRACTORS
COMPETING = ALIGNED + ((0.2, 0.3), (1.2, 0.8), (2.2, 1.3), (3.2, 1.8), (4.2, 2.3), (4.8, 2.6))


def validate_points(points) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("Points must be a finite N×2 array")
    return array


def signed_rho(point, theta_degrees):
    point = np.asarray(point, dtype=float)
    theta = np.asarray(theta_degrees, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all() or not np.isfinite(theta).all():
        raise ValueError("Point and theta must be finite")
    radians = np.deg2rad(theta)
    result = point[0] * np.cos(radians) + point[1] * np.sin(radians)
    return float(result) if np.ndim(result) == 0 else result


def line_from_points(first, second) -> tuple[float, float]:
    """Return signed (rho, theta degrees) for two distinct points."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != (2,) or second.shape != (2,) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Two finite 2D points are required")
    tangent = second - first
    if np.linalg.norm(tangent) <= 1e-12:
        raise ValueError("Coincident points do not determine a line")
    normal = np.array([-tangent[1], tangent[0]])
    theta = float(np.rad2deg(np.arctan2(normal[1], normal[0])) % 180.0)
    return signed_rho(first, theta), theta


def line_segment(rho: float, theta_degrees: float, bounds: tuple[float, float, float, float]):
    """Clip a normal-form line to a rectangle without a slope singularity."""
    xmin, xmax, ymin, ymax = bounds
    theta = np.deg2rad(theta_degrees)
    normal = np.array([np.cos(theta), np.sin(theta)])
    anchor = float(rho) * normal
    tangent = np.array([-normal[1], normal[0]])
    low, high = -np.inf, np.inf
    for axis, minimum, maximum in ((0, xmin, xmax), (1, ymin, ymax)):
        if abs(tangent[axis]) < 1e-12:
            if anchor[axis] < minimum - 1e-10 or anchor[axis] > maximum + 1e-10:
                return None
        else:
            limits = sorted(((minimum - anchor[axis]) / tangent[axis], (maximum - anchor[axis]) / tangent[axis]))
            low, high = max(low, limits[0]), min(high, limits[1])
    if not np.isfinite((low, high)).all() or low > high:
        return None
    return np.vstack((anchor + low * tangent, anchor + high * tangent))


@dataclass(frozen=True)
class Grid:
    theta_edges: np.ndarray
    rho_edges: np.ndarray

    @property
    def theta_centers(self) -> np.ndarray:
        return (self.theta_edges[:-1] + self.theta_edges[1:]) / 2

    @property
    def rho_centers(self) -> np.ndarray:
        return (self.rho_edges[:-1] + self.rho_edges[1:]) / 2

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rho_edges) - 1, len(self.theta_edges) - 1


def make_grid(theta_width: float = 10.0, rho_width: float = 0.5, points=DEFAULT_POINTS) -> Grid:
    if not np.isfinite((theta_width, rho_width)).all() or theta_width <= 0 or rho_width <= 0:
        raise ValueError("Bin widths must be positive and finite")
    if theta_width > 90 or rho_width > 5:
        raise ValueError("Bin widths exceed the teaching UI limit")
    # Preserve the requested width; only the final bin may be shorter.
    theta_edges = np.append(np.arange(0.0, 180.0 - 1e-10, theta_width), 180.0)
    array = validate_points(points)
    # Keep the approved grid for the original fixture; expand symmetrically
    # only when a custom point needs more signed-rho coverage.
    maximum_radius = float(np.linalg.norm(array, axis=1).max(initial=0))
    extent = 5.0 if maximum_radius < 5.0 - 1e-10 else maximum_radius + rho_width
    extent = float(np.ceil(extent / rho_width) * rho_width)
    rho_edges = np.arange(-extent, extent + rho_width * 0.5, rho_width)
    return Grid(theta_edges, rho_edges)


def bin_index(value: float, edges: np.ndarray) -> int:
    if not np.isfinite(value):
        raise ValueError("Non-finite bin coordinate")
    index = int(np.searchsorted(edges, value, side="right") - 1)
    if index < 0 or index >= len(edges) - 1:
        raise ValueError("Vote outside half-open accumulator range")
    return index


def point_votes(point, grid: Grid) -> tuple[tuple[int, int], ...]:
    rho_values = signed_rho(point, grid.theta_centers)
    return tuple((bin_index(float(rho), grid.rho_edges), column) for column, rho in enumerate(rho_values))


def accumulate(points, grid: Grid) -> np.ndarray:
    array = validate_points(points)
    counts = np.zeros(grid.shape, dtype=np.int64)
    for point in array:
        for row, column in point_votes(point, grid):
            counts[row, column] += 1
    return counts


@dataclass(frozen=True)
class VoteState:
    counts: np.ndarray
    next_indices: tuple[int, ...]
    last_vote: tuple[int, int, int] | None = None  # point, rho row, theta column


def empty_votes(points, grid: Grid) -> VoteState:
    return VoteState(np.zeros(grid.shape, dtype=np.int64), (0,) * len(validate_points(points)))


def vote_next(state: VoteState, points, grid: Grid, point_index: int) -> VoteState:
    array = validate_points(points)
    if state.counts.shape != grid.shape or len(state.next_indices) != len(array):
        raise ValueError("Vote state does not match current points/grid")
    if not 0 <= point_index < len(array):
        raise ValueError("Point index outside dataset")
    column = state.next_indices[point_index]
    if column >= grid.shape[1]:
        return state
    row, column = point_votes(array[point_index], grid)[column]
    counts = state.counts.copy()
    counts[row, column] += 1
    indices = list(state.next_indices)
    indices[point_index] += 1
    return VoteState(counts, tuple(indices), (point_index, row, column))


def vote_remaining(state: VoteState, points, grid: Grid, point_index: int) -> VoteState:
    while state.next_indices[point_index] < grid.shape[1]:
        state = vote_next(state, points, grid, point_index)
    return state


def vote_all(state: VoteState, points, grid: Grid) -> VoteState:
    for point_index in range(len(validate_points(points))):
        state = vote_remaining(state, points, grid, point_index)
    return state


def peaks(counts: np.ndarray) -> tuple[int, tuple[tuple[int, int], ...]]:
    maximum = int(counts.max(initial=0))
    if maximum == 0:
        return 0, ()
    return maximum, tuple((int(row), int(column)) for row, column in np.argwhere(counts == maximum))


def peak_geometry(grid: Grid, row: int, column: int) -> tuple[float, float]:
    return float(grid.rho_centers[row]), float(grid.theta_centers[column])
