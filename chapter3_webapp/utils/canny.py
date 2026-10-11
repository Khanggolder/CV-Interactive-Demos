"""NumPy-only teaching Canny, matching the approved Manim prototype."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
KY = KX.T.copy()
PRESETS = (
    "A · Approved scene", "B · Clean geometry", "C · Noisy geometry",
    "D · Connected weak", "E · Isolated weak", "F · Fine detail / blur",
)


def synthetic_input(size: int = 224) -> np.ndarray:
    """The scene's exact seed-13 geometry and uint8 conversion."""
    yy, xx = np.mgrid[:size, :size]
    image = 24 + 18 * (xx / size) + 10 * (yy / size)
    image[(xx - 72) ** 2 + (yy - 80) ** 2 <= 39**2] = 164
    image[(xx > 116) & (xx < 194) & (yy > 104) & (yy < 181)] = 102
    image[np.abs(0.72 * xx + yy - 226) < 4] = 218
    image[np.abs(np.sqrt((xx - 163) ** 2 + (yy - 60) ** 2) - 24) < 3] = 205
    noise = np.random.default_rng(13).normal(0, 13, (size, size))
    texture = 6 * np.sin(xx / 3.7) * np.sin(yy / 5.1)
    return np.clip(image + noise + texture, 0, 255).astype(np.uint8)


def preset_image(name: str) -> np.ndarray:
    if name == PRESETS[0]:
        return synthetic_input()
    n = 128
    yy, xx = np.mgrid[:n, :n]
    image = np.full((n, n), 28.0)
    if name in (PRESETS[1], PRESETS[2]):
        image[(xx - 43)**2 + (yy - 48)**2 <= 25**2] = 190
        image[(xx >= 78) & (xx < 111) & (yy >= 69) & (yy < 105)] = 135
        if name == PRESETS[2]:
            image += np.random.default_rng(29).normal(0, 24, image.shape)
    elif name == PRESETS[3]:
        # A high-contrast segment and its dim continuation share a boundary.
        image[25:105, 30:62] = 160
        image[54:79, 30:62] = np.linspace(160, 68, 25)[:, None]
        image[79:105, 30:62] = 68
        image[25:105, 85:110] = 205
    elif name == PRESETS[4]:
        image[15:57, 15:45] = 195
        image[82:113, 82:110] = 70
    elif name == PRESETS[5]:
        image[19:109, 28:29] = 220
        image[19:109, 35:36] = 220
        image[24:104, 74:77] = 150
    else:
        raise ValueError(f"Unknown preset: {name}")
    return np.clip(image, 0, 255).astype(np.uint8)


def gaussian_kernel(size: int = 5, sigma: float = 1.15) -> np.ndarray:
    if size < 3 or size % 2 != 1 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Kernel size must be odd >=3 and sigma must be positive.")
    axis = np.arange(-(size // 2), size // 2 + 1, dtype=float)
    xx, yy = np.meshgrid(axis, axis)
    weights = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return weights / weights.sum()


def correlate_reflect(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image, kernel = np.asarray(image, dtype=float), np.asarray(kernel, dtype=float)
    if image.ndim != 2 or kernel.ndim != 2 or min(image.shape) < 3:
        raise ValueError("Expected a 2-D image at least 3×3 and a 2-D kernel.")
    py, px = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((py, py), (px, px)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel, optimize=True)


def gradient(smoothed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gx = correlate_reflect(smoothed, KX)
    gy = correlate_reflect(smoothed, KY)
    raw = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)
    return gx, gy, raw, direction


def normalize_magnitude(raw: np.ndarray) -> tuple[np.ndarray, float]:
    scale = max(float(np.percentile(raw, 99.5)), 1e-9)
    return np.clip(raw / scale, 0, 1), scale


def sector(direction: float) -> tuple[int, int, int]:
    angle = float(np.rad2deg(direction)) % 180
    if angle < 22.5 or angle >= 157.5:
        return 0, 1, 0
    if angle < 67.5:
        return 1, 1, 45
    if angle < 112.5:
        return 1, 0, 90
    return 1, -1, 135


def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    if magnitude.shape != direction.shape or magnitude.ndim != 2:
        raise ValueError("Magnitude and direction need matching 2-D shapes.")
    result = np.zeros_like(magnitude, dtype=float)
    angle = np.rad2deg(direction) % 180
    center = magnitude[1:-1, 1:-1]
    sectors = (
        ((angle[1:-1, 1:-1] < 22.5) | (angle[1:-1, 1:-1] >= 157.5), (0, 1)),
        (((angle[1:-1, 1:-1] >= 22.5) & (angle[1:-1, 1:-1] < 67.5)), (1, 1)),
        (((angle[1:-1, 1:-1] >= 67.5) & (angle[1:-1, 1:-1] < 112.5)), (1, 0)),
        (((angle[1:-1, 1:-1] >= 112.5) & (angle[1:-1, 1:-1] < 157.5)), (1, -1)),
    )
    for mask, (dy, dx) in sectors:
        forward = magnitude[1 + dy:magnitude.shape[0] - 1 + dy, 1 + dx:magnitude.shape[1] - 1 + dx]
        backward = magnitude[1 - dy:magnitude.shape[0] - 1 - dy, 1 - dx:magnitude.shape[1] - 1 - dx]
        keep = mask & (center >= forward) & (center >= backward)
        result[1:-1, 1:-1][keep] = center[keep]
    return result


def normalize_nms(nms: np.ndarray) -> np.ndarray:
    return nms / max(float(np.max(nms)), 1e-9)


def inspect_nms(magnitude: np.ndarray, direction: np.ndarray, point: tuple[int, int]) -> dict:
    row, col = point
    if not (1 <= row < magnitude.shape[0] - 1 and 1 <= col < magnitude.shape[1] - 1):
        raise ValueError("NMS inspection requires an interior pixel.")
    dy, dx, degrees = sector(direction[row, col])
    q1, q2 = (row - dy, col - dx), (row + dy, col + dx)
    value = float(magnitude[row, col])
    first, second = float(magnitude[q1]), float(magnitude[q2])
    return dict(point=point, q1=q1, q2=q2, magnitude=value, m1=first, m2=second,
                degrees=degrees, raw_degrees=float(np.rad2deg(direction[row, col])),
                keep=value >= first and value >= second)


def classify(nms_normalized: np.ndarray, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high <= 1):
        raise ValueError("Ngưỡng phải thỏa 0 < low < high ≤ 1.")
    positive = nms_normalized > 0
    strong = positive & (nms_normalized >= high)
    weak = positive & (nms_normalized >= low) & ~strong
    return strong, weak


@dataclass(frozen=True)
class Replay:
    strong: np.ndarray
    weak: np.ndarray
    accepted: np.ndarray
    discovery_step: np.ndarray
    pop_step: np.ndarray
    events: tuple[tuple[tuple[int, int], tuple[tuple[int, int], ...]], ...]


def hysteresis(strong: np.ndarray, weak: np.ndarray) -> Replay:
    if strong.shape != weak.shape or strong.ndim != 2 or np.any(strong & weak):
        raise ValueError("Strong and weak masks must be disjoint, matching 2-D arrays.")
    strong, weak = strong.astype(bool), weak.astype(bool)
    accepted = strong.copy()
    discovered = np.full(strong.shape, -1, dtype=np.int32)
    popped = np.full(strong.shape, -1, dtype=np.int32)
    discovered[strong] = 0
    queue = deque((int(row), int(col)) for row, col in np.argwhere(strong))
    events = []
    height, width = strong.shape
    while queue:
        row, col = queue.popleft()
        step = len(events) + 1
        popped[row, col] = step
        additions = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width and weak[nr, nc] and not accepted[nr, nc]:
                    accepted[nr, nc] = True
                    discovered[nr, nc] = step
                    queue.append((nr, nc))
                    additions.append((nr, nc))
        events.append(((row, col), tuple(additions)))
    return Replay(strong, weak, accepted, discovered, popped, tuple(events))


def replay_frame(replay: Replay, step: int) -> dict:
    if not 0 <= step <= len(replay.events):
        raise ValueError("Replay step outside event range.")
    accepted = replay.strong | (replay.weak & (replay.discovery_step >= 0) & (replay.discovery_step <= step))
    frontier = accepted & ((replay.pop_step < 0) | (replay.pop_step > step))
    active = replay.events[step - 1][0] if step else None
    added = replay.events[step - 1][1] if step else ()
    return dict(accepted=accepted, frontier=frontier, active=active, added=added,
                accepted_weak=int(np.count_nonzero(accepted & replay.weak)))


def base_stages(image: np.ndarray, sigma: float = 1.15) -> dict:
    source = np.asarray(image)
    if source.ndim != 2 or min(source.shape) < 7 or not np.issubdtype(source.dtype, np.number):
        raise ValueError("Expected a grayscale image at least 7×7.")
    if not np.all(np.isfinite(source)) or np.min(source) < 0 or np.max(source) > 255:
        raise ValueError("Image values must be finite intensities in [0,255].")
    original = source.astype(float) / 255.0
    kernel = gaussian_kernel(sigma=sigma)
    smoothed = correlate_reflect(original, kernel)
    gx, gy, raw, direction = gradient(smoothed)
    magnitude, scale = normalize_magnitude(raw)
    nms = non_maximum_suppression(magnitude, direction)
    return dict(original=original, kernel=kernel, smoothed=smoothed, gx=gx, gy=gy,
                magnitude_raw=raw, direction=direction, magnitude=magnitude,
                magnitude_scale=scale, nms=nms, nms_normalized=normalize_nms(nms))


def threshold_stages(base: dict, low: float = 0.10, high: float = 0.35) -> dict:
    strong, weak = classify(base["nms_normalized"], low, high)
    replay = hysteresis(strong, weak)
    return dict(**base, strong=strong, weak=weak, replay=replay, edges=replay.accepted,
                low=low, high=high)


def canny_stages(image: np.ndarray, sigma: float = 1.15, low: float = 0.10,
                 high: float = 0.35) -> dict:
    return threshold_stages(base_stages(image, sigma), low, high)


def nms_examples(arrays: dict) -> dict[str, tuple[int, int] | None]:
    """Find only genuinely present examples, preferring the scene's reference pixels."""
    magnitude, nms, direction = arrays["magnitude"], arrays["nms"], arrays["direction"]
    interior = np.zeros(magnitude.shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    keep = interior & (magnitude > 0.05) & (nms > 0)
    suppress = interior & (magnitude > 0.05) & (nms == 0)
    result = {}
    for name, mask, preferred in (("keep", keep, (130, 116)), ("suppress", suppress, (130, 115))):
        if preferred[0] < mask.shape[0] and preferred[1] < mask.shape[1] and mask[preferred]:
            result[name] = preferred
        else:
            coords = np.argwhere(mask)
            result[name] = tuple(map(int, coords[np.argmax(magnitude[mask])])) if len(coords) else None
    for angle in (0, 45, 90, 135):
        coords = np.argwhere(keep)
        matches = [tuple(map(int, p)) for p in coords if sector(direction[tuple(p)])[2] == angle]
        result[f"sector_{angle}"] = max(matches, key=lambda p: magnitude[p]) if matches else None
    return result


def weak_examples(arrays: dict) -> dict[str, tuple[int, int] | None]:
    weak, accepted = arrays["weak"], arrays["edges"]
    result = {}
    for name, mask in (("connected", weak & accepted), ("isolated", weak & ~accepted)):
        coords = np.argwhere(mask)
        result[name] = tuple(map(int, coords[len(coords) // 2])) if len(coords) else None
    return result
