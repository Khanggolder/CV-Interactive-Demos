"""Shared, Streamlit-free experiment for the three existing Chapter 2 engines.

Arrays are [row, col], displayed points are (x=col, y=row), y down. OpenCV's
2x3 matrix maps source pixel centres to target centres; warpAffine performs
the inverse sampling internally. Truth never participates in descriptor matching.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from utils import harris, orb, sift

METHODS = ("Harris + Normalized Patch Baseline", "SIFT", "ORB-style Educational Pipeline")
SCENES = ("Cấu trúc phân biệt", "Hoa văn lặp", "Đa kích thước", "Bất đối xứng xoay",
          "Thay đổi ánh sáng", "Nhiễu và blur")
PRESETS = {
    SCENES[0]: dict(angle=4., scale=1., dx=3., dy=-2., brightness=0., contrast=1., noise=0., blur=0.),
    SCENES[1]: dict(angle=0., scale=1., dx=9., dy=0., brightness=0., contrast=1., noise=0., blur=0.),
    SCENES[2]: dict(angle=0., scale=.78, dx=0., dy=0., brightness=0., contrast=1., noise=0., blur=0.),
    SCENES[3]: dict(angle=27., scale=1., dx=0., dy=0., brightness=0., contrast=1., noise=0., blur=0.),
    SCENES[4]: dict(angle=0., scale=1., dx=2., dy=0., brightness=.12, contrast=.72, noise=0., blur=0.),
    SCENES[5]: dict(angle=3., scale=1., dx=2., dy=1., brightness=0., contrast=1., noise=.045, blur=1.1),
}


@dataclass(frozen=True)
class Config:
    scene: str = SCENES[0]
    angle: float = 4.
    scale: float = 1.
    dx: float = 3.
    dy: float = -2.
    brightness: float = 0.
    contrast: float = 1.
    noise: float = 0.
    blur: float = 0.
    seed: int = 2026
    budget: int = 30
    rule: str = "Method default"  # or Nearest / Ratio / Cross-check
    ratio: float = .8
    tolerance: float = 4.


@dataclass(frozen=True)
class Pair:
    a: np.ndarray
    b: np.ndarray
    matrix: np.ndarray
    valid_a: np.ndarray
    valid_b: np.ndarray
    config: Config


@dataclass(frozen=True)
class Feature:
    x: float
    y: float
    descriptor: Any
    metadata: dict[str, Any]
    eligible: bool
    location_id: int


@dataclass(frozen=True)
class Detection:
    method: str
    available_locations: int
    retained_locations: int
    orientation_records: int
    features: tuple[Feature, ...]
    seconds: float


@dataclass(frozen=True)
class Decision:
    query: int
    train: int | None
    second: int | None
    distance: float | None
    second_distance: float | None
    accepted: bool
    reason: str
    error: float | None = None
    correct: bool | None = None


@dataclass(frozen=True)
class Result:
    method: str
    pair: Pair
    a: Detection
    b: Detection
    decisions: tuple[Decision, ...]
    rule: str
    match_seconds: float


def generate_challenge_scene(name: str, size: int = 129, seed: int = 2026) -> np.ndarray:
    if name not in SCENES or size < 65 or size > 257 or size % 2 != 1:
        raise ValueError("Unknown scene or unsupported odd image size")
    a = np.full((size, size), .12, np.float64)
    c = size // 2
    if name == SCENES[1]:
        yy, xx = np.indices(a.shape)
        a = np.where(((xx // 14) + (yy // 14)) % 2, .79, .17).astype(float)
        cv2.circle(a, (c + 24, c - 17), 5, .95, -1)
    else:
        cv2.rectangle(a, (16, 19), (49, 53), .81, -1)
        cv2.circle(a, (c + 25, 34), 11, .96, -1)
        cv2.line(a, (c - 20, c + 24), (c + 38, c + 40), .67, 5)
        cv2.rectangle(a, (23, c + 24), (39, c + 43), .41, -1)
        cv2.circle(a, (c + 24, c + 14), 6, .31, -1)
        cv2.line(a, (52, 22), (64, 40), .55, 3)
        if name == SCENES[2]:
            cv2.circle(a, (c - 10, c - 12), 4, .93, -1)
            cv2.circle(a, (c + 6, c + 3), 17, .72, -1)
            cv2.rectangle(a, (18, 74), (25, 81), .94, -1)
        if name == SCENES[3]:
            cv2.line(a, (c + 17, c - 28), (c + 34, c - 10), .44, 4)
            cv2.circle(a, (c - 29, c + 7), 7, .9, -1)
    # Genuine, deterministic low-amplitude texture, not fabricated features.
    rng = np.random.default_rng(seed)
    texture = rng.normal(0, .009, a.shape)
    return np.clip(a + texture, 0, 1)


def map_points_with_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    m = np.asarray(matrix, dtype=float)
    if m.shape != (2, 3) or not np.isfinite(m).all() or not np.isfinite(p).all():
        raise ValueError("Finite Nx2 points and 2x3 affine matrix required")
    return p @ m[:, :2].T + m[:, 2]


def build_transformed_image_pair(config: Config) -> Pair:
    if not (.5 <= config.scale <= 1.5 and .3 <= config.contrast <= 1.7 and
            0 <= config.noise <= .2 and 0 <= config.blur <= 3 and 1 <= config.budget <= 100):
        raise ValueError("Experiment parameters outside safe bounds")
    a = generate_challenge_scene(config.scene, seed=config.seed)
    h, w = a.shape
    matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), config.angle, config.scale)
    matrix[:, 2] += (config.dx, config.dy)
    b = cv2.warpAffine(a, matrix, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Warp an all-one source with the identical interpolation; only fully
    # supported pixels (not partially padded interpolation) count as valid.
    support = cv2.warpAffine(np.ones_like(a), matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid_b = support >= 1 - 1e-7
    if config.blur:
        b = cv2.GaussianBlur(b, (0, 0), config.blur)
    rng = np.random.default_rng(config.seed + 1)
    b = np.clip(config.contrast * b + config.brightness +
                rng.normal(0, config.noise, b.shape), 0, 1)
    return Pair(a, b, matrix, np.ones_like(a, dtype=bool), valid_b, config)


def count_unique_keypoints(points: list[tuple[float, float]], tolerance: float = .75) -> int:
    """Deterministic first-occurrence clusters; SIFT orientations share a location."""
    unique: list[tuple[float, float]] = []
    for x, y in points:
        if np.isfinite((x, y)).all() and all(np.hypot(x-u, y-v) > tolerance for u, v in unique):
            unique.append((x, y))
    return len(unique)


def _location_ids(points: list[tuple[float, float]], tolerance: float = .75) -> list[int]:
    centres: list[tuple[float, float]] = []
    ids = []
    for x, y in points:
        found = next((i for i, (u, v) in enumerate(centres)
                      if np.hypot(x-u, y-v) <= tolerance), None)
        if found is None:
            found = len(centres)
            centres.append((x, y))
        ids.append(found)
    return ids


def _support(mask: np.ndarray, x: float, y: float, radius: int) -> bool:
    col, row = int(np.rint(x)), int(np.rint(y))
    return (row-radius >= 0 and col-radius >= 0 and row+radius < mask.shape[0]
            and col+radius < mask.shape[1]
            and bool(mask[row-radius:row+radius+1, col-radius:col+radius+1].all()))


def normalize_patch_descriptor(patch: np.ndarray) -> np.ndarray:
    p = np.asarray(patch, dtype=np.float32)
    if p.shape != (11, 11) or not np.isfinite(p).all():
        raise ValueError("11x11 finite patch required")
    centred = (p - p.mean()).reshape(-1)
    norm = float(np.linalg.norm(centred))
    return centred / norm if norm > 1e-8 else np.zeros(121, np.float32)


def _detect_harris(image: np.ndarray, mask: np.ndarray, budget: int) -> Detection:
    start = perf_counter()
    response = harris.pipeline(image).harris
    threshold = max(0., float(response.max()) * .008)
    selected = harris.threshold_nms(response, threshold, 5)
    rows, cols = np.nonzero(selected)
    order = sorted(range(len(rows)), key=lambda i: (-response[rows[i], cols[i]], rows[i], cols[i]))
    available = len(order)
    features = []
    for i in order[:budget]:
        y, x = int(rows[i]), int(cols[i])
        eligible = _support(mask, x, y, 5)
        patch = image[y-5:y+6, x-5:x+6]
        descriptor = normalize_patch_descriptor(patch) if eligible else None
        if descriptor is not None and not np.any(descriptor):
            descriptor = None  # a constant patch has no discriminative content
        features.append(Feature(float(x), float(y), descriptor,
                                dict(response=float(response[y, x]), patch=11,
                                     normalization="subtract mean; unit L2; constant -> zero"),
                                eligible, len(features)))
    return Detection(METHODS[0], available, len(features), len(features), tuple(features), perf_counter()-start)


def _detect_sift(image: np.ndarray, mask: np.ndarray, budget: int) -> Detection:
    start = perf_counter()
    pyramid, raw, filtered, _ = sift.pipeline(image, with_descriptors=False)
    ranked = sorted((p for p in filtered if p.status == "kept"),
                    key=lambda p: (-abs(p.response), p.octave, p.row_global, p.col_global))
    available = count_unique_keypoints([(p.col_global, p.row_global) for p in ranked])
    selected = ranked[:budget]
    oriented = sift.describe(pyramid, selected, limit=budget)
    coords = [(p.col_global, p.row_global) for p in oriented]
    ids = _location_ids(coords)
    features = []
    for p, loc_id in zip(oriented, ids):
        # The descriptor uses rotated samples up to about 2.5*sqrt(2)*3*sigma
        # in octave pixels. Reject if the full conservative source footprint
        # touches the transformed padding or original image edge.
        radius = int(np.ceil(11 * p.sigma_global))
        eligible = _support(mask, p.col_global, p.row_global, radius)
        descriptor = p.descriptor.astype(np.float32) if eligible and p.descriptor is not None else None
        features.append(Feature(float(p.col_global), float(p.row_global), descriptor,
                                dict(octave=p.octave, sigma=p.sigma_global,
                                     orientation=p.orientation, response=p.response,
                                     support_radius=radius), eligible, loc_id))
    return Detection(METHODS[1], available,
                     count_unique_keypoints([(p.col_global, p.row_global) for p in selected]),
                     len(features), tuple(features), perf_counter()-start)


def _detect_orb(image: np.ndarray, mask: np.ndarray, budget: int) -> Detection:
    start = perf_counter()
    pyramid, points, _ = orb.orb_pipeline(image*255, max_per_level=25)
    # Preserve native per-level FAST/NMS/Harris ranking, then distribute budget
    # across levels instead of comparing incomparable scores across scales.
    by_level: dict[int, list] = {}
    for p in points:
        by_level.setdefault(p.level, []).append(p)
    selected = []
    while len(selected) < budget and any(by_level.values()):
        for level in sorted(by_level):
            if by_level[level] and len(selected) < budget:
                selected.append(by_level[level].pop(0))
    coords = [(p.x_original, p.y_original) for p in selected]
    ids = _location_ids(coords)
    features = []
    for p, loc_id in zip(selected, ids):
        # Radius 9 centroid plus blurred/steered BRIEF support 7 + Gaussian
        # blur footprint ~5 in level pixels, mapped conservatively to source.
        level = pyramid[p.level]
        radius = int(np.ceil(13 * max(level.scale_x, level.scale_y)))
        eligible = _support(mask, p.x_original, p.y_original, radius)
        descriptor = p.descriptor if eligible and p.descriptor is not None else None
        features.append(Feature(float(p.x_original), float(p.y_original), descriptor,
                                dict(level=p.level, angle=p.angle, fast_score=p.fast_score,
                                     harris_score=p.harris_score, support_radius=radius,
                                     pattern="deterministic educational, not OpenCV learned"),
                                eligible, loc_id))
    return Detection(METHODS[2], count_unique_keypoints([(p.x_original,p.y_original) for p in points]),
                     count_unique_keypoints(coords), len(features), tuple(features), perf_counter()-start)


def detect_pair(pair: Pair, method: str) -> tuple[Detection, Detection]:
    if method not in METHODS:
        raise ValueError("Unknown method")
    fn = (_detect_harris, _detect_sift, _detect_orb)[METHODS.index(method)]
    a = fn(pair.a, pair.valid_a, pair.config.budget)
    extra = int(np.ceil(4*pair.config.blur)) if pair.config.blur else 0
    target_mask = pair.valid_b
    if extra:
        target_mask = cv2.erode(target_mask.astype(np.uint8),
                                np.ones((2*extra+1, 2*extra+1), np.uint8),
                                borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    b = fn(pair.b, target_mask, pair.config.budget)
    # Source feature must have observable target support as well; this is an
    # eligibility filter, not a descriptor-match or correctness preselection.
    updated = []
    for f in a.features:
        expected = map_points_with_affine([[f.x, f.y]], pair.matrix)[0]
        radius = int(f.metadata.get("support_radius", 5))
        overlap = _support(target_mask, *expected, radius)
        updated.append(replace(f, eligible=f.eligible and overlap,
                               descriptor=f.descriptor if f.eligible and overlap else None))
    return replace(a, features=tuple(updated)), b


def _distance(method: str, a: Any, b: Any) -> float:
    return float(orb.calculate_hamming_distance(a, b)) if method == METHODS[2] else float(np.linalg.norm(a-b))


def match_features(a: Detection, b: Detection, rule: str = "Method default",
                   ratio: float = .8) -> tuple[Decision, ...]:
    if not 0 < ratio < 1 or rule not in ("Method default", "Nearest", "Ratio", "Cross-check"):
        raise ValueError("Invalid matching rule")
    actual = ("Cross-check" if a.method == METHODS[2] else "Ratio") if rule == "Method default" else rule
    queries = [(i, f) for i, f in enumerate(a.features) if f.descriptor is not None]
    targets = [(j, f) for j, f in enumerate(b.features) if f.descriptor is not None]
    reverse = {}
    if actual == "Cross-check":
        for j, target in targets:
            reverse[j] = min(((_distance(a.method, target.descriptor, q.descriptor), i)
                              for i, q in queries), default=(float("inf"), None))[1]
    out = []
    for i, q in queries:
        ranked = sorted((_distance(a.method, q.descriptor, f.descriptor), j)
                        for j, f in targets)
        if not ranked:
            out.append(Decision(i, None, None, None, None, False, "no target descriptor"))
            continue
        d1, j = ranked[0]
        d2, k = ranked[1] if len(ranked) > 1 else (None, None)
        if actual == "Ratio":
            # Reuse the SIFT engine's strict Lowe decision for 128D vectors.
            if a.method == METHODS[1]:
                native = sift.apply_lowe_ratio_test(np.array([d for d, _ in ranked]), ratio)
                accepted = native.accepted
            else:
                accepted = d2 is not None and d2 > 0 and d1/d2 < ratio
            reason = "ratio pass" if accepted else "ratio reject / insufficient distinct neighbors"
        elif actual == "Cross-check":
            accepted = reverse[j] == i
            reason = "mutual nearest" if accepted else "not mutual"
        else:
            accepted = True
            reason = "nearest"
        out.append(Decision(i, j, k, d1, d2, bool(accepted), reason))
    return tuple(out)


def evaluate_match_geometry(pair: Pair, a: Detection, b: Detection,
                            decisions: tuple[Decision, ...], tolerance: float = 4.) -> tuple[Decision, ...]:
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    result = []
    for d in decisions:
        if not d.accepted or d.train is None:
            result.append(d)
            continue
        source, target = a.features[d.query], b.features[d.train]
        expected = map_points_with_affine([[source.x, source.y]], pair.matrix)[0]
        error = float(np.hypot(target.x-expected[0], target.y-expected[1]))
        result.append(replace(d, error=error, correct=bool(error <= tolerance)))
    return tuple(result)


def run_method(pair: Pair, method: str) -> Result:
    a, b = detect_pair(pair, method)
    start = perf_counter()
    decisions = match_features(a, b, pair.config.rule, pair.config.ratio)
    seconds = perf_counter()-start
    return Result(method, pair, a, b,
                  evaluate_match_geometry(pair, a, b, decisions, pair.config.tolerance),
                  pair.config.rule, seconds)


def compute_matching_metrics(result: Result) -> dict[str, Any]:
    accepted = [d for d in result.decisions if d.accepted]
    evaluable = [d for d in accepted if d.error is not None and np.isfinite(d.error)]
    correct = [d for d in evaluable if d.correct]
    source_locations = {f.location_id for f in result.a.features if f.descriptor is not None}
    covered = {result.a.features[d.query].location_id for d in correct}
    errors = [d.error for d in evaluable]
    return dict(unique_a=result.a.retained_locations, unique_b=result.b.retained_locations,
                available_a=result.a.available_locations, available_b=result.b.available_locations,
                records_a=result.a.orientation_records, records_b=result.b.orientation_records,
                descriptors_a=sum(f.descriptor is not None for f in result.a.features),
                descriptors_b=sum(f.descriptor is not None for f in result.b.features),
                candidates=len(result.decisions), accepted=len(accepted), evaluable=len(evaluable),
                correct=len(correct), incorrect=len(evaluable)-len(correct),
                precision=len(correct)/len(evaluable) if evaluable else None,
                coverage=len(covered)/len(source_locations) if source_locations else None,
                coverage_numerator=len(covered), coverage_denominator=len(source_locations),
                median_error=float(np.median(errors)) if errors else None,
                p90_error=float(np.percentile(errors, 90)) if errors else None,
                detection_seconds=result.a.seconds+result.b.seconds,
                matching_seconds=result.match_seconds,
                descriptor_bytes=484 if result.method == METHODS[0] else (512 if result.method == METHODS[1] else 32),
                rule=result.rule)


def generate_experiment_snapshot(results: list[Result], prediction: str = "",
                                 explanation: str = "") -> dict[str, Any]:
    if not results:
        raise ValueError("At least one measured result required")
    config = results[0].pair.config
    return dict(id=f"{config.scene}-{config.seed}-{config.angle:g}-{config.scale:g}",
                config=dict(config.__dict__), prediction=prediction, explanation=explanation,
                metrics={r.method: compute_matching_metrics(r) for r in results})
