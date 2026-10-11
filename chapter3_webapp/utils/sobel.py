"""Pure Sobel arithmetic. Displayed masks are correlated, never flipped.

Rows/y increase downward; columns/x increase rightward. Only valid 3x3
interior centers are computed; map borders are NaN (undefined), not padded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int64)
KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int64)
DEFAULT_CENTER = (2, 3)
PRESET_NAMES = (
    "A · Vertical edge", "B · Horizontal edge", "C · Diagonal edge",
    "D · Flat region", "E · Reversed contrast", "F · Noisy edge",
)


def validate_grayscale_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2 or min(array.shape) < 3:
        raise ValueError("Ảnh grayscale phải có ít nhất 3×3 pixel.")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Intensity phải là số nguyên 0–255.")
    if np.any(array < 0) or np.any(array > 255):
        raise ValueError("Intensity phải nằm trong 0–255.")
    return array.astype(np.int64, copy=True)


def valid_center(image: np.ndarray, center: tuple[int, int]) -> bool:
    array = validate_grayscale_image(image)
    row, col = center
    return (isinstance(row, (int, np.integer)) and isinstance(col, (int, np.integer))
            and 1 <= row < array.shape[0] - 1 and 1 <= col < array.shape[1] - 1)


def extract_patch(image: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    array = validate_grayscale_image(image)
    if not valid_center(array, center):
        raise ValueError("Kernel center phải nằm trong vùng interior hợp lệ.")
    row, col = center
    return array[row - 1:row + 2, col - 1:col + 2].copy()


@dataclass(frozen=True)
class SobelResult:
    patch: np.ndarray
    products_x: np.ndarray
    products_y: np.ndarray
    row_sums_x: np.ndarray
    row_sums_y: np.ndarray
    gx: int
    gy: int
    magnitude: float
    theta_degrees: float | None


def compute_sobel_at(image: np.ndarray, center: tuple[int, int]) -> SobelResult:
    patch = extract_patch(image, center)
    px, py = patch * KX, patch * KY  # int64: no uint8 wraparound
    gx, gy = int(px.sum()), int(py.sum())
    magnitude = float(np.hypot(gx, gy))
    theta = float(np.degrees(np.arctan2(gy, gx))) if magnitude else None
    return SobelResult(patch, px, py, px.sum(axis=1), py.sum(axis=1),
                       gx, gy, magnitude, theta)


def compute_gradient_maps(image: np.ndarray) -> dict[str, np.ndarray]:
    array = validate_grayscale_image(image)
    maps = {name: np.full(array.shape, np.nan) for name in ("Gx", "Gy", "Magnitude", "Direction")}
    for row in range(1, array.shape[0] - 1):
        for col in range(1, array.shape[1] - 1):
            result = compute_sobel_at(array, (row, col))
            maps["Gx"][row, col] = result.gx
            maps["Gy"][row, col] = result.gy
            maps["Magnitude"][row, col] = result.magnitude
            if result.theta_degrees is not None:
                maps["Direction"][row, col] = result.theta_degrees
    return maps


def screen_vector(gx: int, gy: int) -> tuple[int, int]:
    return (int(gx), -int(gy))


def normalize_for_visualization(values: np.ndarray, *, signed: bool = False) -> np.ndarray:
    """Return display-only 0..1 values; retain NaN border/undefined cells."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    scale = float(np.max(np.abs(finite))) if finite.size else 0.0
    if signed:
        return 0.5 + np.divide(array, 2 * scale) if scale else np.where(np.isfinite(array), 0.5, np.nan)
    if np.any(finite < 0):
        raise ValueError("Unsigned display requires nonnegative data.")
    return array / scale if scale else np.where(np.isfinite(array), 0.0, np.nan)


def edit_pixel(image: np.ndarray, row: int, col: int, value: int) -> np.ndarray:
    array = validate_grayscale_image(image)
    if not (0 <= row < array.shape[0] and 0 <= col < array.shape[1]):
        raise ValueError("Pixel index ngoài ảnh.")
    if not isinstance(value, (int, np.integer)) or not 0 <= value <= 255:
        raise ValueError("Intensity phải là số nguyên 0–255.")
    array[row, col] = value
    return array


def create_sobel_presets() -> dict[str, np.ndarray]:
    vertical = np.full((5, 7), 20, dtype=np.int64)
    vertical[:, 4:] = 200
    horizontal = np.full((5, 7), 20, dtype=np.int64)
    horizontal[3:, :] = 200
    diagonal = np.full((5, 7), 20, dtype=np.int64)
    for row in range(5):
        diagonal[row, max(0, 5 - row):] = 200
    reversed_edge = np.full((5, 7), 200, dtype=np.int64)
    reversed_edge[:, 4:] = 20
    rng = np.random.default_rng(37)
    noisy = np.clip(vertical + rng.integers(-45, 46, vertical.shape), 0, 255)
    return dict(zip(PRESET_NAMES, (vertical, horizontal, diagonal,
                                   np.full((5, 7), 100, dtype=np.int64), reversed_edge, noisy)))
