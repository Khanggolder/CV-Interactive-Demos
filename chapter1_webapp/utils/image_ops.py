"""Small, side-effect-free image operations used by the Streamlit UI."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Return a clipped uint8 copy suitable for display and OpenCV."""
    return np.clip(image, 0, 255).astype(np.uint8)


def load_rgb_image(source: object, max_size: int = 1600) -> np.ndarray:
    """Decode a path or uploaded file as a bounded RGB uint8 array."""
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8).copy()


def to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale without mutating the input."""
    if image_rgb.ndim == 2:
        return image_rgb.copy()
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 1:
        return image_rgb[:, :, 0].copy()
    return cv2.cvtColor(ensure_uint8(image_rgb), cv2.COLOR_RGB2GRAY)


def mark_pixel_and_patch(
    image: np.ndarray, x: int, y: int, patch_radius: int = 3
) -> np.ndarray:
    """Return an RGB copy with the selected pixel and patch visibly marked."""
    base = ensure_uint8(image)
    marked = (
        cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
        if base.ndim == 2
        else base.copy()
    )
    height, width = marked.shape[:2]
    x0, x1 = max(0, x - patch_radius), min(width - 1, x + patch_radius)
    y0, y1 = max(0, y - patch_radius), min(height - 1, y + patch_radius)
    thickness = max(2, min(height, width) // 220)
    marker_size = max(15, min(height, width) // 24)
    cv2.rectangle(marked, (x0, y0), (x1, y1), (255, 180, 0), thickness)
    cv2.drawMarker(
        marked,
        (x, y),
        (255, 35, 35),
        markerType=cv2.MARKER_CROSS,
        markerSize=marker_size,
        thickness=thickness,
    )
    return marked


def adjust_brightness_contrast(
    image: np.ndarray, alpha: float = 1.0, beta: int = 0
) -> np.ndarray:
    """Apply g = alpha * I + beta and clip the result to [0, 255]."""
    transformed = alpha * image.astype(np.float32) + float(beta)
    return ensure_uint8(transformed)


def gamma_correct(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply g = 255 * (I / 255) ** gamma."""
    normalized = image.astype(np.float32) / 255.0
    return ensure_uint8(255.0 * np.power(normalized, gamma))


def equalize_histogram(image_rgb: np.ndarray) -> np.ndarray:
    """Equalize grayscale directly or only the luminance of an RGB image."""
    if image_rgb.ndim == 2:
        return cv2.equalizeHist(ensure_uint8(image_rgb))
    ycrcb = cv2.cvtColor(ensure_uint8(image_rgb), cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def apply_clahe(
    image_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8
) -> np.ndarray:
    """Apply CLAHE to grayscale or to the luminance of an RGB image."""
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    if image_rgb.ndim == 2:
        return clahe.apply(ensure_uint8(image_rgb))
    ycrcb = cv2.cvtColor(ensure_uint8(image_rgb), cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
