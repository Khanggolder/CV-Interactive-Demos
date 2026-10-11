"""Pure Cartesian circle geometry and finite, deterministic Hough voting.

Image and center coordinates both use y-up. Accumulator A[r, b_row, a_col]
uses half-open center/radius bins. Each observation supports a cell at most once
per radius layer, regardless of angular oversampling or polarity duplicates.
This is a teaching detector, separate from the approved continuous Manim scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CENTER = (1.5, 0.5)
RADIUS = 2.0
ANGLES = (180, 60, 300, 0, 120, 240)
EDGE_WIDTH = 0.15
MAX_POINTS = 30
MAX_LAYERS = 16
MAX_SAMPLES = 180
MAX_CELLS = 240_000  # int32 accumulator <= 960,000 bytes


def _xy(value) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError("Cần tọa độ 2D hữu hạn")
    return result


def _radius(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Bán kính phải hữu hạn và dương")
    return value


def circle_point(center, radius: float, phi_degrees):
    center, radius = _xy(center), _radius(radius)
    phi = np.asarray(phi_degrees, dtype=float)
    if not np.isfinite(phi).all():
        raise ValueError("Góc phải hữu hạn")
    radians = np.deg2rad(phi)
    result = center + radius * np.stack((np.cos(radians), np.sin(radians)), axis=-1)
    return result


def candidate_center(point, radius: float, phi_degrees):
    return circle_point(point, radius, phi_degrees)


def center_locus(point, radius: float, samples: int = 241) -> np.ndarray:
    if not 4 <= samples <= 2000:
        raise ValueError("Số điểm vẽ locus ngoài giới hạn")
    return circle_point(point, radius, np.linspace(0, 360, samples))


def circle_circle_intersections(first, second, first_radius: float, second_radius: float,
                                tolerance: float = 1e-10) -> tuple[np.ndarray, ...]:
    p, q = _xy(first), _xy(second)
    r, s = _radius(first_radius), _radius(second_radius)
    delta = q - p
    distance = float(np.linalg.norm(delta))
    if distance <= tolerance:
        if abs(r - s) <= tolerance:
            raise ValueError("Hai locus trùng nhau: vô số tâm khả dĩ")
        return ()
    if distance > r + s + tolerance or distance < abs(r - s) - tolerance:
        return ()
    along = (r*r - s*s + distance*distance) / (2*distance)
    height_squared = r*r - along*along
    if height_squared < -tolerance:
        return ()
    foot = p + (along / distance) * delta
    if height_squared <= tolerance:
        return (foot,)
    offset = np.sqrt(height_squared) * np.array([-delta[1], delta[0]]) / distance
    return (foot + offset, foot - offset)


def common_centers(points, radius: float, tolerance: float = 1e-8) -> tuple[np.ndarray, ...]:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("Cần mảng điểm N×2 hữu hạn")
    if len(array) < 2:
        return ()
    for i in range(1, len(array)):
        if np.linalg.norm(array[i] - array[0]) > tolerance:
            candidates = circle_circle_intersections(array[0], array[i], radius, radius)
            return tuple(c for c in candidates
                         if np.all(np.abs(np.linalg.norm(array - c, axis=1) - radius) <= tolerance))
    raise ValueError("Các điểm trùng nhau: tâm chưa xác định")


def circle_from_three_points(points) -> tuple[np.ndarray, float]:
    p = np.asarray(points, dtype=float)
    if p.shape != (3, 2) or not np.isfinite(p).all():
        raise ValueError("Cần đúng ba điểm hữu hạn")
    matrix = 2 * (p[1:] - p[0])
    if abs(np.linalg.det(matrix)) <= 1e-12:
        raise ValueError("Ba điểm trùng/thẳng hàng không xác định đường tròn duy nhất")
    rhs = np.sum(p[1:]**2, axis=1) - np.sum(p[0]**2)
    center = np.linalg.solve(matrix, rhs)
    return center, float(np.linalg.norm(p[0] - center))


def synthetic_intensity(xy, center=CENTER, radius=RADIUS, *, bright_inside=True):
    xy = np.asarray(xy, dtype=float)
    distance = np.linalg.norm(xy - _xy(center), axis=-1)
    value = .5 + .5 * np.tanh((_radius(radius) - distance) / EDGE_WIDTH)
    return value if bright_inside else 1 - value


def synthetic_gradient(xy, center=CENTER, radius=RADIUS, *, bright_inside=True):
    xy = np.asarray(xy, dtype=float)
    delta = xy - _xy(center)
    distance = np.linalg.norm(delta, axis=-1)
    if np.any(distance <= 1e-12):
        raise ValueError("Gradient hướng bán kính không xác định tại tâm")
    factor = -.5 / EDGE_WIDTH * (1 - np.tanh((_radius(radius) - distance) / EDGE_WIDTH)**2)
    result = factor[..., None] * delta / distance[..., None]
    return result if bright_inside else -result


def gradient_candidates(point, radius: float, gradient, polarity: str) -> tuple[np.ndarray, ...]:
    point, radius = _xy(point), _radius(radius)
    gradient = _xy(gradient)
    norm = float(np.linalg.norm(gradient))
    if norm <= 1e-12:
        raise ValueError("Gradient zero/không tin cậy: không thể định hướng vote")
    direction = gradient / norm
    if polarity == "inward":
        return (point + radius * direction,)
    if polarity == "outward":
        return (point - radius * direction,)
    if polarity == "unknown":
        return (point + radius * direction, point - radius * direction)
    raise ValueError("Polarity không hợp lệ")


@dataclass(frozen=True)
class Observation:
    id: str
    xy: tuple[float, float]
    gradient: tuple[float, float] | None = None
    reference: bool = True


@dataclass(frozen=True)
class Dataset:
    name: str
    points: tuple[Observation, ...]
    true_circles: tuple[tuple[float, float, float], ...]
    question: str
    limitation: str


def generate_circle_example() -> Dataset:
    points = circle_point(CENTER, RADIUS, ANGLES)
    gradients = synthetic_gradient(points)
    observations = tuple(Observation(f"p{i+1}", tuple(map(float, p)), tuple(map(float, g)))
                         for i, (p, g) in enumerate(zip(points, gradients)))
    return Dataset("A · Vòng tròn chuẩn", observations, ((*CENTER, RADIUS),),
                   "Hai điểm có cho đúng một tâm?", "Giao loci liên tục khác peak trên lưới hữu hạn.")


def preset_dataset(name: str) -> Dataset:
    base = generate_circle_example()
    if name == "A · Vòng tròn chuẩn":
        return base
    if name == "B · Chưa biết bán kính":
        return Dataset(name, base.points, base.true_circles,
                       "r=1, 2 hay 3 tạo peak tốt hơn?", "Không tiết lộ r trước khi thăm các lớp.")
    if name == "C · Cung bị thiếu":
        return Dataset(name, base.points[:3], base.true_circles,
                       "Ít điểm trên một cung còn bao nhiêu bằng chứng?", "Ít support hơn; không mặc định thất bại.")
    if name == "D · Có outlier":
        extra = ((-1.8, -1.5), (4.8, 3.4), (0.1, 3.6), (5.0, -1.8))
        pts = base.points + tuple(Observation(f"o{i+1}", p) for i, p in enumerate(extra))
        return Dataset(name, pts, base.true_circles,
                       "Outlier có thể tạo peak cạnh tranh?", "Outlier không có gradient: full-locus mới dùng được.")
    if name == "E · Đảo polarity":
        pts = tuple(Observation(p.id, p.xy, tuple(-v for v in p.gradient)) for p in base.points)
        return Dataset(name, pts, base.true_circles,
                       "Gradient đảo chiều thì dấu vote phải đổi ra sao?", "Sai polarity làm mất tâm thật.")
    if name == "F · Hai vòng tròn":
        center, radius = (4.1, 2.6), 1.35
        more = circle_point(center, radius, (20, 80, 140, 200, 260, 320))
        gradient = synthetic_gradient(more, center, radius)
        pts = base.points + tuple(Observation(f"q{i+1}", tuple(map(float, p)), tuple(map(float, g)))
                                  for i, (p, g) in enumerate(zip(more, gradient)))
        return Dataset(name, pts, ((*CENTER, RADIUS), (*center, radius)),
                       "Một peak cao nhất có mô tả đủ hai vòng tròn?", "Cần xem nhiều giả thuyết; peak thứ hai có thể bị che.")
    raise ValueError("Preset không hợp lệ")


PRESETS = ("A · Vòng tròn chuẩn", "B · Chưa biết bán kính", "C · Cung bị thiếu",
           "D · Có outlier", "E · Đảo polarity", "F · Hai vòng tròn")


def validate_dataset(dataset: Dataset) -> None:
    if not 1 <= len(dataset.points) <= MAX_POINTS:
        raise ValueError(f"Giới hạn {MAX_POINTS} quan sát")
    ids = [p.id for p in dataset.points]
    if len(set(ids)) != len(ids):
        raise ValueError("ID quan sát phải duy nhất")
    for point in dataset.points:
        _xy(point.xy)
        if point.gradient is not None:
            _xy(point.gradient)


@dataclass(frozen=True)
class Grid:
    a_edges: np.ndarray
    b_edges: np.ndarray
    r_edges: np.ndarray
    fixed_radius: bool = False

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.r_edges)-1, len(self.b_edges)-1, len(self.a_edges)-1

    @property
    def a_centers(self):
        return (self.a_edges[:-1] + self.a_edges[1:]) / 2

    @property
    def b_centers(self):
        return (self.b_edges[:-1] + self.b_edges[1:]) / 2

    @property
    def r_centers(self):
        return (self.r_edges[:-1] + self.r_edges[1:]) / 2


def _edges(low: float, high: float, width: float) -> np.ndarray:
    if not np.isfinite((low, high, width)).all() or high <= low or width <= 0:
        raise ValueError("Bounds/bin width không hợp lệ")
    count = int(np.ceil((high-low) / width))
    if count > MAX_CELLS:
        raise ValueError("Quá nhiều bin")
    return np.r_[low + np.arange(count) * width, high].astype(float)


def estimate_accumulator_memory(shape) -> int:
    dimensions = tuple(int(v) for v in shape)
    if len(dimensions) != 3 or any(v <= 0 for v in dimensions):
        raise ValueError("Shape phải là (r,b,a) dương")
    # Plain Python integers avoid int64 overflow for a rejected huge request.
    return dimensions[0] * dimensions[1] * dimensions[2] * np.dtype(np.int32).itemsize


def make_grid(*, center_bounds=(-4.0, 7.0, -4.0, 6.0), center_width=.4,
              radius=2.0, radius_range=None, radius_width=.5) -> Grid:
    amin, amax, bmin, bmax = map(float, center_bounds)
    a_edges, b_edges = _edges(amin, amax, center_width), _edges(bmin, bmax, center_width)
    fixed = radius_range is None
    if fixed:
        radius = _radius(radius)
        r_edges = np.array([radius-radius_width/2, radius+radius_width/2], dtype=float)
        if r_edges[0] < 0 or not np.isfinite(r_edges).all():
            raise ValueError("Radius bin không hợp lệ")
    else:
        low, high = map(float, radius_range)
        _radius(low)
        r_edges = _edges(low, high, radius_width)
    shape = (len(r_edges)-1, len(b_edges)-1, len(a_edges)-1)
    if shape[0] > MAX_LAYERS or np.prod(shape) > MAX_CELLS:
        raise ValueError(f"Lưới quá lớn: {shape}; tối đa {MAX_LAYERS} lớp, {MAX_CELLS} cells")
    return Grid(a_edges, b_edges, r_edges, fixed)


def bin_index(value: float, edges: np.ndarray) -> int | None:
    if not np.isfinite(value):
        raise ValueError("Vote không hữu hạn")
    index = int(np.searchsorted(edges, value, side="right") - 1)
    return index if 0 <= index < len(edges)-1 else None


@dataclass(frozen=True)
class VoteEvent:
    point_index: int
    radius_index: int
    sample_index: int
    candidate: tuple[float, float]
    cell: tuple[int, int, int] | None
    counted: bool


def vote_events(dataset: Dataset, grid: Grid, *, mode="full", angular_samples=72,
                polarity="inward") -> tuple[tuple[VoteEvent, ...], ...]:
    validate_dataset(dataset)
    if not 4 <= angular_samples <= MAX_SAMPLES:
        raise ValueError(f"Angular samples phải trong [4,{MAX_SAMPLES}]")
    if mode not in ("full", "gradient", "unknown"):
        raise ValueError("Voting mode không hợp lệ")
    per_point = []
    seen_coordinates = set()
    for point_index, obs in enumerate(dataset.points):
        # Equal observed coordinates are one geometric observation, even if IDs differ.
        coordinate = tuple(obs.xy)
        duplicate = coordinate in seen_coordinates
        seen_coordinates.add(coordinate)
        record = []
        for radius_index, radius in enumerate(grid.r_centers):
            if mode == "full":
                candidates = circle_point(obs.xy, float(radius), np.arange(angular_samples)*360/angular_samples)
            elif obs.gradient is None:
                candidates = np.empty((0, 2))
            else:
                try:
                    candidates = np.asarray(gradient_candidates(obs.xy, float(radius), obs.gradient,
                                                   "unknown" if mode == "unknown" else polarity))
                except ValueError:
                    candidates = np.empty((0, 2))
            seen_cells = set()
            for sample_index, candidate in enumerate(candidates):
                a_col = bin_index(float(candidate[0]), grid.a_edges)
                b_row = bin_index(float(candidate[1]), grid.b_edges)
                cell = None if a_col is None or b_row is None else (radius_index, b_row, a_col)
                counted = bool(cell is not None and cell not in seen_cells and not duplicate)
                if cell is not None:
                    seen_cells.add(cell)
                record.append(VoteEvent(point_index, radius_index, sample_index,
                                        tuple(map(float, candidate)), cell, counted))
        per_point.append(tuple(record))
    return tuple(per_point)


@dataclass(frozen=True)
class VoteState:
    counts: np.ndarray
    next_indices: tuple[int, ...]
    signature: tuple
    last_event: VoteEvent | None = None


def vote_signature(dataset: Dataset, grid: Grid, mode: str, angular_samples: int, polarity: str) -> tuple:
    return (tuple((p.id, p.xy, p.gradient) for p in dataset.points),
            tuple(grid.a_edges), tuple(grid.b_edges), tuple(grid.r_edges),
            mode, angular_samples, polarity)


def empty_votes(dataset: Dataset, grid: Grid, mode="full", angular_samples=72,
                polarity="inward") -> VoteState:
    vote_events(dataset, grid, mode=mode, angular_samples=angular_samples, polarity=polarity)
    return VoteState(np.zeros(grid.shape, dtype=np.int32), (0,) * len(dataset.points),
                     vote_signature(dataset, grid, mode, angular_samples, polarity))


def _check(state, dataset, grid, mode, angular_samples, polarity):
    if state.signature != vote_signature(dataset, grid, mode, angular_samples, polarity):
        raise ValueError("Vote state thuộc cấu hình cũ; cần reset")


def vote_next(state: VoteState, dataset: Dataset, grid: Grid, point_index: int,
              *, mode="full", angular_samples=72, polarity="inward") -> VoteState:
    _check(state, dataset, grid, mode, angular_samples, polarity)
    events = vote_events(dataset, grid, mode=mode, angular_samples=angular_samples, polarity=polarity)
    if not 0 <= point_index < len(events):
        raise ValueError("Point index ngoài phạm vi")
    index = state.next_indices[point_index]
    if index >= len(events[point_index]):
        return state
    event = events[point_index][index]
    counts = state.counts.copy()
    if event.counted:
        counts[event.cell] += 1
    positions = list(state.next_indices)
    positions[point_index] += 1
    return VoteState(counts, tuple(positions), state.signature, event)


def _from_positions(state, events, positions):
    counts = np.zeros_like(state.counts)
    for point_events, length in zip(events, positions):
        for event in point_events[:length]:
            if event.counted:
                counts[event.cell] += 1
    return VoteState(counts, tuple(positions), state.signature)


def vote_remaining(state, dataset, grid, point_index, *, mode="full", angular_samples=72,
                   polarity="inward"):
    _check(state, dataset, grid, mode, angular_samples, polarity)
    events = vote_events(dataset, grid, mode=mode, angular_samples=angular_samples, polarity=polarity)
    positions = list(state.next_indices)
    positions[point_index] = len(events[point_index])
    return _from_positions(state, events, positions)


def vote_all(state, dataset, grid, *, mode="full", angular_samples=72, polarity="inward"):
    _check(state, dataset, grid, mode, angular_samples, polarity)
    events = vote_events(dataset, grid, mode=mode, angular_samples=angular_samples, polarity=polarity)
    return _from_positions(state, events, [len(record) for record in events])


def vote_previous(state, dataset, grid, point_index, *, mode="full", angular_samples=72,
                  polarity="inward"):
    _check(state, dataset, grid, mode, angular_samples, polarity)
    events = vote_events(dataset, grid, mode=mode, angular_samples=angular_samples, polarity=polarity)
    positions = list(state.next_indices)
    positions[point_index] = max(0, positions[point_index]-1)
    return _from_positions(state, events, positions)


def peaks(counts: np.ndarray) -> tuple[int, tuple[tuple[int, int, int], ...]]:
    if counts.ndim != 3:
        raise ValueError("Accumulator phải có shape (r,b,a)")
    maximum = int(counts.max(initial=0))
    if maximum == 0:
        return 0, ()
    cells = np.argwhere(counts == maximum)
    return maximum, tuple(tuple(map(int, cell)) for cell in cells)


def top_cells(counts: np.ndarray, limit=10) -> tuple[tuple[int, tuple[int, int, int]], ...]:
    cells = np.argwhere(counts > 0)
    ranked = sorted(((int(counts[tuple(cell)]), tuple(map(int, cell))) for cell in cells),
                    key=lambda item: (-item[0], item[1]))
    return tuple(ranked[:limit])


def cell_parameters(grid: Grid, cell: tuple[int, int, int]) -> tuple[float, float, float]:
    layer, row, col = cell
    return float(grid.a_centers[col]), float(grid.b_centers[row]), float(grid.r_centers[layer])
