"""Numerically explicit RANSAC line fitting used by the teaching UI.

Lines use the normalized implicit form ``a*x + b*y + c = 0``.  Keeping the
implicit form avoids a slope special case and makes perpendicular distances
and vertical lines behave exactly like every other line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TOLERANCE = 1e-12


class RansacError(ValueError):
    """Base exception for invalid numerical inputs."""


class DegenerateSampleError(RansacError):
    """Raised when points cannot define a unique line."""


@dataclass(frozen=True)
class GeneratedDataset:
    """A generated point cloud and labels used only to validate generation."""

    points: np.ndarray
    generated_inlier_labels: np.ndarray


@dataclass(frozen=True)
class Trial:
    """One computed RANSAC hypothesis."""

    iteration: int
    pair: tuple[int, int]
    model: np.ndarray
    distances: np.ndarray
    inlier_mask: np.ndarray
    score: int
    inlier_error: float


@dataclass(frozen=True)
class RansacResult:
    """The best hypothesis and its TLS refit, if enough consensus exists."""

    trials: tuple[Trial, ...]
    best_trial: Trial | None
    refit_model: np.ndarray | None
    refit_mask: np.ndarray | None


def validate_points(points: np.ndarray, minimum: int = 1) -> np.ndarray:
    """Return finite ``N x 2`` points as float64 or raise a clear error."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise RansacError("points must have shape (N, 2)")
    if len(array) < minimum:
        raise RansacError(f"at least {minimum} point(s) are required")
    if not np.isfinite(array).all():
        raise RansacError("points must contain only finite values")
    return array


def normalize_line(coefficients: np.ndarray | list[float]) -> np.ndarray:
    """Normalize ``[a,b,c]`` so ``sqrt(a²+b²)=1`` with a canonical sign."""
    model = np.asarray(coefficients, dtype=np.float64).copy()
    if model.shape != (3,) or not np.isfinite(model).all():
        raise RansacError("line coefficients must be three finite values")
    length = float(np.linalg.norm(model[:2]))
    if length <= TOLERANCE:
        raise DegenerateSampleError("line normal has zero length")
    model /= length
    if model[1] < -TOLERANCE or (
        abs(model[1]) <= TOLERANCE and model[0] < 0
    ):
        model *= -1.0
    return model


def line_from_two_points(point_1: np.ndarray, point_2: np.ndarray) -> np.ndarray:
    """Fit the unique implicit line through two distinct finite points."""
    pair = validate_points(np.vstack((point_1, point_2)), minimum=2)
    delta = pair[1] - pair[0]
    if float(np.linalg.norm(delta)) <= TOLERANCE:
        raise DegenerateSampleError("sampled points are duplicates")
    normal = np.array([-delta[1], delta[0]], dtype=np.float64)
    return normalize_line([normal[0], normal[1], -normal @ pair[0]])


def signed_distances(points: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Signed perpendicular distances to a normalized implicit line."""
    array = validate_points(points)
    line = normalize_line(model)
    return array @ line[:2] + line[2]


def perpendicular_distances(points: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Absolute perpendicular distance ``|a*x+b*y+c|/sqrt(a²+b²)``."""
    return np.abs(signed_distances(points, model))


def classify_inliers(
    points: np.ndarray, model: np.ndarray, epsilon: float
) -> np.ndarray:
    """Classify points using the inclusive rule ``distance <= epsilon``."""
    if not np.isfinite(epsilon) or epsilon < 0:
        raise RansacError("epsilon must be finite and non-negative")
    # Floating-point arithmetic can put a sampled point ~1e-16 away from its
    # own line. Treat that round-off as zero, including when epsilon is zero.
    return perpendicular_distances(points, model) <= float(epsilon) + TOLERANCE


def consensus_score(points: np.ndarray, model: np.ndarray, epsilon: float) -> int:
    """Count consensus inliers for one model."""
    return int(classify_inliers(points, model, epsilon).sum())


def perpendicular_feet(points: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Return the perpendicular projection of each point onto the line."""
    array = validate_points(points)
    line = normalize_line(model)
    signed = array @ line[:2] + line[2]
    return array - signed[:, None] * line[:2]


def fit_tls(points: np.ndarray) -> np.ndarray:
    """Total least-squares/PCA line minimizing squared perpendicular error."""
    array = validate_points(points, minimum=2)
    center = array.mean(axis=0)
    centered = array - center
    if float(np.linalg.norm(centered)) <= TOLERANCE:
        raise DegenerateSampleError("consensus points do not span a line")
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) == 0 or singular_values[0] <= TOLERANCE:
        raise DegenerateSampleError("consensus points do not span a line")
    normal = vh[-1]
    return normalize_line([normal[0], normal[1], -normal @ center])


def fit_ordinary_least_squares(points: np.ndarray) -> np.ndarray:
    """Fit vertical-residual OLS ``y=m*x+k`` with a vertical-data fallback.

    Ordinary y-on-x regression is undefined when all x coordinates coincide.
    For that edge case the only meaningful visible model is ``x=mean(x)``.
    """
    array = validate_points(points, minimum=2)
    if float(np.ptp(array[:, 0])) <= TOLERANCE:
        return normalize_line([1.0, 0.0, -float(array[:, 0].mean())])
    design = np.column_stack((array[:, 0], np.ones(len(array))))
    slope, intercept = np.linalg.lstsq(design, array[:, 1], rcond=None)[0]
    return normalize_line([-slope, 1.0, -intercept])


def evaluate_pair(
    points: np.ndarray,
    pair: tuple[int, int] | np.ndarray,
    epsilon: float,
    iteration: int = 0,
) -> Trial:
    """Fit and score one explicitly selected pair."""
    array = validate_points(points, minimum=2)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise RansacError("epsilon must be finite and non-negative")
    indices = tuple(int(value) for value in pair)
    if len(indices) != 2 or indices[0] == indices[1]:
        raise DegenerateSampleError("a sample needs two different indices")
    if min(indices) < 0 or max(indices) >= len(array):
        raise RansacError("sample index is outside the point cloud")
    model = line_from_two_points(array[indices[0]], array[indices[1]])
    distances = perpendicular_distances(array, model)
    mask = distances <= float(epsilon) + TOLERANCE
    error = float(np.square(distances[mask]).sum()) if mask.any() else float("inf")
    return Trial(
        iteration=int(iteration),
        pair=indices,
        model=model,
        distances=distances,
        inlier_mask=mask,
        score=int(mask.sum()),
        inlier_error=error,
    )


def deterministic_pairs(
    points: np.ndarray, seed: int, count: int, max_draw_multiplier: int = 100
) -> list[tuple[int, int]]:
    """Draw reproducible valid pairs, skipping duplicate-coordinate samples."""
    array = validate_points(points, minimum=2)
    if count < 0:
        raise RansacError("count must be non-negative")
    if count == 0:
        return []
    rng = np.random.default_rng(int(seed))
    pairs: list[tuple[int, int]] = []
    max_draws = max(1, count * max_draw_multiplier)
    for _ in range(max_draws):
        raw = rng.choice(len(array), size=2, replace=False)
        pair = (int(raw[0]), int(raw[1]))
        if float(np.linalg.norm(array[pair[1]] - array[pair[0]])) <= TOLERANCE:
            continue
        pairs.append(pair)
        if len(pairs) == count:
            return pairs
    if len(pairs) < count:
        raise DegenerateSampleError("no valid pair can be sampled from these points")
    return pairs


def run_ransac(
    points: np.ndarray, epsilon: float, seed: int, iterations: int
) -> RansacResult:
    """Run deterministic RANSAC and TLS-refit the winning consensus set."""
    array = validate_points(points, minimum=2)
    if iterations < 0:
        raise RansacError("iterations must be non-negative")
    pairs = deterministic_pairs(array, seed, iterations)
    trials = tuple(
        evaluate_pair(array, pair, epsilon, index + 1)
        for index, pair in enumerate(pairs)
    )
    if not trials:
        return RansacResult(trials, None, None, None)
    # Larger consensus wins.  The residual sum is only a deterministic tie-break.
    best = min(trials, key=lambda trial: (-trial.score, trial.inlier_error, trial.iteration))
    if best.score < 2:
        return RansacResult(trials, best, None, None)
    try:
        refit = fit_tls(array[best.inlier_mask])
    except DegenerateSampleError:
        refit = None
    refit_mask = (
        classify_inliers(array, refit, epsilon) if refit is not None else None
    )
    return RansacResult(trials, best, refit, refit_mask)


def generate_dataset(
    number_inliers: int,
    number_outliers: int,
    noise: float,
    seed: int,
    slope: float = 0.65,
    intercept: float = 0.3,
    x_limits: tuple[float, float] = (-4.0, 4.0),
    y_limits: tuple[float, float] = (-3.2, 3.2),
) -> GeneratedDataset:
    """Generate a reproducible teaching dataset without using labels for fitting."""
    if number_inliers < 0 or number_outliers < 0:
        raise RansacError("point counts must be non-negative")
    if number_inliers + number_outliers < 2:
        raise RansacError("the dataset needs at least two points")
    if noise < 0 or not np.isfinite(noise):
        raise RansacError("noise must be finite and non-negative")
    rng = np.random.default_rng(int(seed))
    x_values = np.linspace(x_limits[0] + 0.35, x_limits[1] - 0.35, number_inliers)
    rng.shuffle(x_values)
    inliers = np.column_stack(
        (x_values, slope * x_values + intercept + rng.normal(0, noise, number_inliers))
    )
    outliers = np.column_stack(
        (
            rng.uniform(*x_limits, number_outliers),
            rng.uniform(*y_limits, number_outliers),
        )
    )
    points = np.vstack((inliers, outliers)).astype(np.float64)
    labels = np.arange(len(points)) < number_inliers
    permutation = rng.permutation(len(points))
    return GeneratedDataset(points[permutation], labels[permutation])


def guided_dataset() -> GeneratedDataset:
    """Return the deterministic 20-point cloud used in the approved animation."""
    rng = np.random.default_rng(37)
    x_values = np.linspace(-3.4, 3.4, 14)
    inliers = np.column_stack(
        (x_values, 0.65 * x_values + 0.3 + rng.normal(0, 0.095, len(x_values)))
    )
    outliers = np.array(
        [
            [-3.8, 2.6],
            [-2.5, 2.5],
            [-0.7, -2.6],
            [1.0, -2.3],
            [3.3, -2.7],
            [4.0, -2.4],
        ],
        dtype=np.float64,
    )
    return GeneratedDataset(
        np.vstack((inliers, outliers)), np.arange(20, dtype=int) < 14
    )


def line_equation(model: np.ndarray, decimals: int = 3) -> str:
    """Format a normalized implicit model without assuming a finite slope."""
    a, b, c = normalize_line(model)
    return f"{a:.{decimals}f}x {b:+.{decimals}f}y {c:+.{decimals}f} = 0"
