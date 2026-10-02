"""Homogeneous transforms and explicit inverse-warping helpers."""

from __future__ import annotations

import cv2
import numpy as np


def translation_matrix(tx: float, ty: float) -> np.ndarray:
    """Return a forward homogeneous translation matrix in (x, y) coordinates."""
    return np.array(
        [[1.0, 0.0, float(tx)], [0.0, 1.0, float(ty)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def centered_rotation_scale_matrix(
    shape: tuple[int, int], *, angle_degrees: float = 0.0, scale: float = 1.0
) -> np.ndarray:
    """Compose a forward rotation/scale around the image center."""
    height, width = shape
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    theta = np.deg2rad(float(angle_degrees))
    cosine = float(np.cos(theta)) * float(scale)
    sine = float(np.sin(theta)) * float(scale)

    # With y increasing downward, this matrix matches a visually positive
    # (counter-clockwise) OpenCV rotation.
    rotation_scale = np.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    to_origin = translation_matrix(-center_x, -center_y)
    back_to_center = translation_matrix(center_x, center_y)
    return back_to_center @ rotation_scale @ to_origin


def build_transform_matrix(
    shape: tuple[int, int],
    operation: str,
    *,
    angle_degrees: float = 0.0,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
) -> np.ndarray:
    """Build the forward matrix used by the selected teaching operation."""
    if operation == "Rotate":
        return centered_rotation_scale_matrix(shape, angle_degrees=angle_degrees)
    if operation == "Scale":
        return centered_rotation_scale_matrix(shape, scale=scale)
    if operation == "Translate":
        return translation_matrix(tx, ty)
    raise ValueError(f"Unsupported operation: {operation}")


def inverse_source_coordinate(
    matrix: np.ndarray, output_x: float, output_y: float
) -> tuple[float, float]:
    """Map one destination coordinate back to its real-valued source point."""
    inverse = np.linalg.inv(matrix)
    source = inverse @ np.array([output_x, output_y, 1.0], dtype=np.float64)
    source /= source[2]
    return float(source[0]), float(source[1])


def inverse_coordinate_maps(
    shape: tuple[int, int], matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build source-x/source-y maps for every destination pixel."""
    height, width = shape
    output_y, output_x = np.indices((height, width), dtype=np.float64)
    destinations = np.stack(
        (output_x.ravel(), output_y.ravel(), np.ones(height * width)), axis=0
    )
    sources = np.linalg.inv(matrix) @ destinations
    sources /= sources[2]
    source_x = sources[0].reshape(height, width).astype(np.float32)
    source_y = sources[1].reshape(height, width).astype(np.float32)
    return source_x, source_y


def warp_inverse(
    image: np.ndarray, matrix: np.ndarray, interpolation: str
) -> np.ndarray:
    """Warp by explicitly mapping each destination pixel to the source image."""
    source_x, source_y = inverse_coordinate_maps(image.shape[:2], matrix)
    flags = {
        "Nearest Neighbor": cv2.INTER_NEAREST,
        "Bilinear": cv2.INTER_LINEAR,
    }
    if interpolation not in flags:
        raise ValueError(f"Unsupported interpolation: {interpolation}")
    return cv2.remap(
        image,
        source_x,
        source_y,
        flags[interpolation],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def nearest_sample(
    image: np.ndarray, source_x: float, source_y: float
) -> tuple[tuple[int, int], np.ndarray | np.uint8 | None]:
    """Return the nearest integer source coordinate and its value, if valid."""
    x_nearest = int(np.floor(source_x + 0.5))
    y_nearest = int(np.floor(source_y + 0.5))
    height, width = image.shape[:2]
    if not (0 <= x_nearest < width and 0 <= y_nearest < height):
        return (x_nearest, y_nearest), None
    return (x_nearest, y_nearest), image[y_nearest, x_nearest].copy()


def bilinear_sample_details(
    image: np.ndarray, source_x: float, source_y: float
) -> dict[str, object]:
    """Calculate the exact four-neighbor bilinear interpolation for one point."""
    x0 = int(np.floor(source_x))
    y0 = int(np.floor(source_y))
    x1, y1 = x0 + 1, y0 + 1
    dx, dy = float(source_x - x0), float(source_y - y0)
    height, width = image.shape[:2]
    channels = () if image.ndim == 2 else (image.shape[2],)

    def pixel(x: int, y: int) -> np.ndarray:
        if 0 <= x < width and 0 <= y < height:
            return np.asarray(image[y, x], dtype=np.float64)
        return np.zeros(channels, dtype=np.float64)

    q11 = pixel(x0, y0)
    q21 = pixel(x1, y0)
    q12 = pixel(x0, y1)
    q22 = pixel(x1, y1)
    r1 = (1.0 - dx) * q11 + dx * q21
    r2 = (1.0 - dx) * q12 + dx * q22
    value = (1.0 - dy) * r1 + dy * r2
    return {
        "coordinates": ((x0, y0), (x1, y0), (x0, y1), (x1, y1)),
        "values": (q11, q21, q12, q22),
        "dx": dx,
        "dy": dy,
        "r1": r1,
        "r2": r2,
        "value": value,
    }


def forward_mapping_occupancy(
    shape: tuple[int, int], matrix: np.ndarray
) -> np.ndarray:
    """Mark destination cells hit by nearest-neighbor forward mapping."""
    height, width = shape
    source_y, source_x = np.indices((height, width), dtype=np.float64)
    sources = np.stack(
        (source_x.ravel(), source_y.ravel(), np.ones(height * width)), axis=0
    )
    destinations = matrix @ sources
    destinations /= destinations[2]
    x_dest = np.floor(destinations[0] + 0.5).astype(int)
    y_dest = np.floor(destinations[1] + 0.5).astype(int)
    valid = (
        (x_dest >= 0) & (x_dest < width) & (y_dest >= 0) & (y_dest < height)
    )
    occupancy = np.zeros((height, width), dtype=np.uint8)
    occupancy[y_dest[valid], x_dest[valid]] = 255
    return occupancy
