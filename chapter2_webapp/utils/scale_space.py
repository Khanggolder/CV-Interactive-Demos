"""Discrete scale-space teaching engine; arrays are float64, indexed [row, col].

Synthetic inputs are assumed to have zero *known* preblur. GaussianBlur uses a
separable, normalized, finite kernel (radius ceil(4 sigma)), REFLECT_101 borders.
Pyramid octave o uses pixels spaced 2**o original pixels apart. Downsampling
takes every second pixel from Gaussian level s, already anti-aliased by sigma
1.6*2 in the previous octave. No extra blur is applied at that transition.
The continuous semigroup and LoG/DoG identity are only approximate for finite
sampled kernels. LoG here is the five-point finite-difference Laplacian of L.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import cv2
import numpy as np

PRESET_NAMES = ("Ba blob sáng", "Một blob sáng", "Blob tối", "Blob có nhiễu",
                "Texture đa thang", "Góc chữ L", "Dải sọc", "Ảnh phẳng")
PRESET_QUESTIONS = (
    "Blob nào biến mất trước khi tăng σ?", "Đỉnh đáp ứng xuất hiện ở σ nào?",
    "Đổi polarity thì dấu LoG/DoG ra sao?", "Nhiễu còn mạnh ở σ lớn không?",
    "Ô vuông nhỏ và lớn mất chi tiết theo thứ tự nào?", "Cửa sổ Harris cố định có đủ cho mọi tỉ lệ?",
    "Dải sọc tạo blob thật hay ứng viên không mong muốn?", "Ảnh phẳng có cực trị strict không?",
)


def _image(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 9 or max(a.shape) > 512 or not np.isfinite(a).all():
        raise ValueError("Expected finite 2D image, 9..512 pixels per side")
    return a


def presets(size: int = 129) -> dict[str, np.ndarray]:
    if size < 65 or size > 257 or size % 2 != 1:
        raise ValueError("Preset size must be odd and 65..257")
    y, x = np.indices((size, size))
    c = size // 2
    disk = lambda row, col, radius: ((x-col)**2 + (y-row)**2 <= radius**2).astype(float)
    one = disk(c, c, 11)
    three = np.clip(disk(c-28, c-29, 4) + disk(c+23, c-4, 9)
                    + disk(c-8, c+31, 16), 0, 1)
    rng = np.random.default_rng(2026)
    corner = ((x >= c) & (y >= c)).astype(float)
    return dict(zip(PRESET_NAMES, (
        three, one, 1-one, np.clip(one+rng.normal(0, .08, one.shape), 0, 1),
        (((x//5 + y//5) % 2)*.45 + ((x//19 + y//19) % 2)*.45),
        corner, ((x//7) % 2).astype(float), np.full((size, size), .4))))


def gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    if not np.isfinite(sigma) or sigma <= 0 or not 2 <= truncate <= 6:
        raise ValueError("sigma must be positive; truncate in [2,6]")
    radius = max(1, ceil(truncate*sigma))
    axis = np.arange(-radius, radius+1, dtype=float)
    one = np.exp(-axis**2/(2*sigma**2))
    one /= one.sum()
    return np.outer(one, one)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    a = _image(image)
    kernel = gaussian_kernel(sigma)
    side = kernel.shape[0]
    return cv2.GaussianBlur(a, (side, side), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT_101)


def incremental_sigma(previous: float, target: float) -> float:
    if not np.isfinite(previous) or not np.isfinite(target) or previous < 0 or target <= previous:
        raise ValueError("Require finite 0 <= previous < target")
    return sqrt(target**2-previous**2)


def semigroup(image: np.ndarray, sigma1: float, sigma2: float) -> dict[str, object]:
    delta = incremental_sigma(sigma1, sigma2)
    direct = gaussian_blur(image, sigma2)
    first = gaussian_blur(image, sigma1)
    incremental = gaussian_blur(first, delta)
    wrong = gaussian_blur(first, sigma2-sigma1)
    difference = np.abs(direct-incremental)
    return {"direct": direct, "incremental": incremental, "wrong": wrong,
            "difference": difference, "mae": float(difference.mean()),
            "max_error": float(difference.max()), "delta": delta,
            "wrong_mae": float(np.mean(np.abs(direct-wrong)))}


@dataclass(frozen=True)
class Octave:
    index: int
    gaussians: tuple[np.ndarray, ...]
    dogs: tuple[np.ndarray, ...]
    local_sigmas: tuple[float, ...]
    global_sigmas: tuple[float, ...]
    increments: tuple[float, ...]


def build_pyramid(image: np.ndarray, octaves: int = 3, intervals: int = 3,
                  base_sigma: float = 1.6, input_sigma: float = 0.0) -> tuple[Octave, ...]:
    """input_sigma is known blur in original-image pixels, not octave-local pixels."""
    a = _image(image)
    if not 1 <= octaves <= 4 or not 2 <= intervals <= 5 or not 1 <= base_sigma <= 2.5:
        raise ValueError("Unsupported pyramid dimensions or sigma")
    if not 0 <= input_sigma < base_sigma:
        raise ValueError("Input blur must be known, nonnegative and below base sigma")
    if min(a.shape)//(2**(octaves-1)) < 16:
        raise ValueError("Last octave must be at least 16 pixels per side")
    k = 2**(1/intervals)
    base = gaussian_blur(a, incremental_sigma(input_sigma, base_sigma))
    output = []
    for octave_index in range(octaves):
        sigmas = tuple(base_sigma*k**j for j in range(intervals+3))
        levels = [base]
        increments = [incremental_sigma(input_sigma, base_sigma) if octave_index == 0 else 0.0]
        for j in range(1, intervals+3):
            inc = incremental_sigma(sigmas[j-1], sigmas[j])
            levels.append(gaussian_blur(levels[-1], inc))
            increments.append(inc)
        dogs = tuple(levels[j+1]-levels[j] for j in range(intervals+2))
        output.append(Octave(octave_index, tuple(levels), dogs, sigmas,
                             tuple(s*2**octave_index for s in sigmas), tuple(increments)))
        if octave_index+1 < octaves:
            base = levels[intervals][::2, ::2].copy()
    return tuple(output)


def log_response(image: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Five-point discrete Laplacian after Gaussian smoothing, REFLECT_101 border."""
    smooth = gaussian_blur(image, sigma)
    lap, normalized = log_from_level(smooth, sigma)
    return smooth, lap, normalized


def log_from_level(gaussian_level: np.ndarray, sigma_local: float) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference LoG of an existing Gaussian level at its *total* local sigma."""
    a = _image(gaussian_level)
    if not np.isfinite(sigma_local) or sigma_local <= 0:
        raise ValueError("sigma_local must be positive")
    lap = cv2.Laplacian(a, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_REFLECT_101)
    return lap, sigma_local**2*lap


def dog(image_low: np.ndarray, image_high: np.ndarray) -> np.ndarray:
    a, b = _image(image_low), _image(image_high)
    if a.shape != b.shape:
        raise ValueError("Gaussian levels must have the same shape")
    return b-a


def log_dog_comparison(image: np.ndarray, sigma: float, k: float) -> dict[str, object]:
    if not np.isfinite(k) or k <= 1 or k > 2:
        raise ValueError("Require 1 < k <= 2")
    _, lap, normalized = log_response(image, sigma)
    low = gaussian_blur(image, sigma)
    high = gaussian_blur(image, sigma*k)
    actual = dog(low, high)
    comparable = actual/(k-1)
    residual = comparable-normalized
    return {"low": low, "high": high, "lap": lap, "normalized": normalized,
            "dog": actual, "comparable": comparable, "residual": residual,
            "mae": float(np.mean(np.abs(residual)))}


@dataclass(frozen=True)
class NeighborEvent:
    layer: int
    row: int
    col: int
    value: float
    max_possible: bool
    min_possible: bool


def inspect_neighbors(stack: tuple[np.ndarray, ...] | list[np.ndarray], layer: int,
                      row: int, col: int) -> tuple[NeighborEvent, ...]:
    if layer <= 0 or layer >= len(stack)-1:
        raise ValueError("Candidate must have lower and upper DoG scales")
    current = _image(stack[layer])
    if any(_image(a).shape != current.shape for a in stack):
        raise ValueError("DoG layers must match dimensions")
    if not 1 <= row < current.shape[0]-1 or not 1 <= col < current.shape[1]-1:
        raise ValueError("Candidate must have complete 3x3x3 neighborhood")
    center = float(current[row, col])
    events = []
    max_ok = min_ok = True
    for scale in (layer-1, layer, layer+1):
        for rr in range(row-1, row+2):
            for cc in range(col-1, col+2):
                if (scale, rr, cc) == (layer, row, col):
                    continue
                value = float(stack[scale][rr, cc])
                max_ok &= center > value
                min_ok &= center < value
                events.append(NeighborEvent(scale, rr, cc, value, max_ok, min_ok))
    assert len(events) == 26
    return tuple(events)


def replay(events: tuple[NeighborEvent, ...], count: int) -> str:
    if count < 0 or count > len(events):
        raise ValueError("Invalid event prefix")
    if count == 0:
        return "Đang kiểm tra"
    event = events[count-1]
    if not event.max_possible and not event.min_possible:
        return "Bị loại"
    if count < len(events):
        return "Còn khả năng MAX" if event.max_possible else "Còn khả năng MIN"
    return "MAX" if event.max_possible else "MIN"


def detect_extrema(stack: tuple[np.ndarray, ...] | list[np.ndarray]) -> list[tuple[int, int, int, float, str]]:
    """Strict extrema only. Equal-valued plateaus and incomplete borders are rejected."""
    if len(stack) < 3:
        return []
    result = []
    for layer in range(1, len(stack)-1):
        a = _image(stack[layer])
        for row in range(1, a.shape[0]-1):
            for col in range(1, a.shape[1]-1):
                value = float(a[row, col])
                # Fast rejection avoids allocating 26 events for almost every pixel.
                blocks = [stack[j][row-1:row+2, col-1:col+2] for j in (layer-1,layer,layer+1)]
                mid = np.delete(blocks[1].ravel(), 4)
                neighbors = np.concatenate((blocks[0].ravel(), mid, blocks[2].ravel()))
                if np.all(value > neighbors):
                    result.append((layer, row, col, value, "MAX"))
                elif np.all(value < neighbors):
                    result.append((layer, row, col, value, "MIN"))
    return result


def filter_contrast(candidates: list[tuple[int, int, int, float, str]], threshold: float):
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Contrast threshold must be nonnegative")
    return [item for item in candidates if abs(item[3]) >= threshold]


def map_octave(row: int, col: int, octave: int) -> tuple[int, int]:
    if row < 0 or col < 0 or octave < 0 or octave > 3:
        raise ValueError("Invalid octave coordinate")
    return row*2**octave, col*2**octave


def scale_curve(image: np.ndarray, row: int, col: int,
                sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = _image(image)
    if not 0 <= row < a.shape[0] or not 0 <= col < a.shape[1]:
        raise ValueError("Pixel out of bounds")
    raw, normalized = [], []
    for sigma in sigmas:
        _, lap, norm = log_response(a, float(sigma))
        raw.append(float(lap[row,col]))
        normalized.append(float(norm[row,col]))
    return np.array(raw), np.array(normalized)
