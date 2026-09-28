"""Fourier-transform operations kept separate from presentation code."""

from __future__ import annotations

import numpy as np


def compute_fft(image_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the raw 2-D FFT and its zero-centered version."""
    fft = np.fft.fft2(image_gray.astype(np.float32))
    return fft, np.fft.fftshift(fft)


def magnitude_spectrum(shifted_fft: np.ndarray) -> np.ndarray:
    """Return a display-normalized log magnitude spectrum."""
    magnitude = np.log1p(np.abs(shifted_fft))
    low, high = float(magnitude.min()), float(magnitude.max())
    if high <= low:
        return np.zeros_like(magnitude, dtype=np.uint8)
    return ((magnitude - low) * 255.0 / (high - low)).astype(np.uint8)


def linear_magnitude_spectrum(shifted_fft: np.ndarray) -> np.ndarray:
    """Return a linearly normalized magnitude image (before log compression)."""
    magnitude = np.abs(shifted_fft)
    low, high = float(magnitude.min()), float(magnitude.max())
    if high <= low:
        return np.zeros_like(magnitude, dtype=np.uint8)
    return ((magnitude - low) * 255.0 / (high - low)).astype(np.uint8)


def create_lowpass_mask(shape: tuple[int, int], radius: int) -> np.ndarray:
    """Create a centered binary ideal low-pass mask."""
    rows, cols = shape
    yy, xx = np.ogrid[:rows, :cols]
    distance_squared = (yy - rows // 2) ** 2 + (xx - cols // 2) ** 2
    return (distance_squared <= radius**2).astype(np.float32)


def create_highpass_mask(shape: tuple[int, int], radius: int) -> np.ndarray:
    """Create a centered binary ideal high-pass mask."""
    return 1.0 - create_lowpass_mask(shape, radius)


def apply_frequency_mask(
    shifted_fft: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Multiply a shifted spectrum by a frequency mask."""
    return shifted_fft * mask


def reconstruct_image(masked_shifted_fft: np.ndarray) -> np.ndarray:
    """Return the real-valued inverse FFT without display normalization."""
    reconstructed = np.real(np.fft.ifft2(np.fft.ifftshift(masked_shifted_fft)))
    return reconstructed.astype(np.float32)


def reconstruction_for_display(
    reconstructed: np.ndarray, *, signed: bool = False
) -> np.ndarray:
    """Prepare an IFFT result for display while preserving its mathematical data.

    Low-pass results use physical clipping only. Signed high-pass responses use a
    symmetric display mapping where zero becomes mid-gray; this mapping is for
    visualization and must not be reused as image data.
    """
    if not signed:
        return np.clip(reconstructed, 0, 255).astype(np.uint8)

    max_abs = float(np.max(np.abs(reconstructed)))
    if max_abs == 0:
        return np.full(reconstructed.shape, 128, dtype=np.uint8)
    display = 127.5 + 127.5 * reconstructed / max_abs
    return np.rint(np.clip(display, 0, 255)).astype(np.uint8)
