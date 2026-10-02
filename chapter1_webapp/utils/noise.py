"""Deterministic noise generation and filtering helpers."""

from __future__ import annotations

import cv2
import numpy as np


def add_gaussian_noise(
    image: np.ndarray, sigma: float, seed: int
) -> np.ndarray:
    """Add zero-mean Gaussian noise without mutating ``image``."""
    source = image.astype(np.float32)
    if sigma <= 0:
        return image.copy()
    rng = np.random.default_rng(seed)
    noisy = source + rng.normal(0.0, float(sigma), size=image.shape)
    return np.rint(np.clip(noisy, 0, 255)).astype(np.uint8)


def add_salt_pepper_noise(
    image: np.ndarray,
    density: float,
    seed: int,
    salt_ratio: float = 0.5,
) -> np.ndarray:
    """Replace a deterministic subset of pixels with black or white impulses."""
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be in [0, 1]")
    if not 0.0 <= salt_ratio <= 1.0:
        raise ValueError("salt_ratio must be in [0, 1]")

    noisy = image.copy()
    if density == 0:
        return noisy

    rng = np.random.default_rng(seed)
    selector = rng.random(image.shape[:2])
    pepper = selector < density * (1.0 - salt_ratio)
    salt = (selector >= density * (1.0 - salt_ratio)) & (selector < density)
    noisy[pepper] = 0
    noisy[salt] = 255
    return noisy


def gaussian_filter(
    image: np.ndarray, kernel_size: int, sigma: float
) -> np.ndarray:
    """Apply a Gaussian smoothing filter with a reflected image boundary."""
    _validate_odd_kernel(kernel_size)
    return cv2.GaussianBlur(
        image,
        (int(kernel_size), int(kernel_size)),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )


def median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply the nonlinear median neighborhood filter."""
    _validate_odd_kernel(kernel_size)
    return cv2.medianBlur(image, int(kernel_size))


def _validate_odd_kernel(kernel_size: int) -> None:
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")
