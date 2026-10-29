"""Deterministic camera fixture and cached measurements for SIFT Deep Dive V2.

All pixel, scale, keypoint, histogram, descriptor, and matching values come from
the approved Chapter 2 numerical engines. This module does not change them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from chapter2_common import signed_rgba
from utils import scale_space, sift


def camera_image(size: int = 129) -> np.ndarray:
    """A small asymmetric camera with lens, viewfinder, flash, dial and detail."""
    if size != 129:
        raise ValueError("The verified teaching fixture is 129 by 129")
    a = np.full((size, size), 0.10, dtype=np.float64)
    cv2.rectangle(a, (18, 36), (111, 94), 0.68, -1)
    cv2.rectangle(a, (24, 30), (53, 37), 0.82, -1)
    cv2.rectangle(a, (92, 31), (105, 36), 0.92, -1)
    cv2.circle(a, (65, 65), 25, 0.26, -1)
    cv2.circle(a, (65, 65), 19, 0.88, -1)
    cv2.circle(a, (65, 65), 13, 0.31, -1)
    cv2.circle(a, (65, 65), 6, 0.91, -1)
    cv2.circle(a, (99, 48), 5, 0.94, -1)
    cv2.rectangle(a, (25, 75), (38, 86), 0.24, -1)
    cv2.rectangle(a, (43, 78), (52, 82), 0.90, -1)
    cv2.line(a, (93, 78), (105, 84), 0.23, 3)
    cv2.circle(a, (35, 48), 3, 0.94, -1)
    return a


@dataclass(frozen=True)
class DeepDiveData:
    image: np.ndarray
    transformed: np.ndarray
    affine: np.ndarray
    pyramid: tuple
    raw: tuple
    filtered: tuple
    described: tuple
    pyramid_b: tuple
    raw_b: tuple
    filtered_b: tuple
    described_b: tuple
    candidate: sift.Keypoint
    selected: sift.Keypoint
    oriented: sift.Keypoint
    orientation: sift.Orientation
    descriptor: sift.Descriptor
    accepted: sift.Match
    rejected: sift.Match | None
    geometry_error: float


@lru_cache(maxsize=1)
def build_data() -> DeepDiveData:
    image = camera_image()
    transformed, affine = sift.transform_scene(image, rotation=17, scale=0.94)
    pyramid, raw, filtered, described = sift.pipeline(image)
    pyramid_b, raw_b, filtered_b, described_b = sift.pipeline(transformed)
    survivors = [p for p in filtered if p.status == "kept"]
    if not survivors or not described or len(described_b) < 2:
        raise ValueError("Camera fixture has insufficient SIFT features")
    matches = sift.match_descriptors(described, described_b)
    accepted = [m for m in matches if m.accepted]
    if not accepted:
        raise ValueError("Camera fixture has no accepted descriptor match")
    # Select a real accepted match with a low post-hoc geometric error. The
    # affine matrix does not participate in descriptor distance or ratio tests.
    ranked = [(sift.evaluate_geometric_correspondence(described[m.query],
               described_b[m.nearest], affine)[0], m) for m in accepted]
    error, best = min(ranked, key=lambda item: item[0])
    rejected = next((m for m in matches if m.ratio is not None and not m.accepted), None)
    moving = [p for p in survivors if p.octave == 0 and
              abs(p.delta[0]) + abs(p.delta[1]) > 0.15 and
              sift.calculate_orientation_histogram(p, pyramid[0]).peaks]
    selected = max(moving, key=lambda p: abs(p.response))
    candidate = raw[next(i for i, p in enumerate(filtered) if p is selected)]
    orientation = sift.calculate_orientation_histogram(selected, pyramid[selected.octave])
    oriented = sift.assign_keypoint_orientations(selected, pyramid[selected.octave])[0]
    descriptor = sift.build_sift_descriptor(oriented, pyramid[oriented.octave])
    return DeepDiveData(image, transformed, affine, pyramid, tuple(raw), tuple(filtered),
                        tuple(described), pyramid_b, tuple(raw_b), tuple(filtered_b),
                        tuple(described_b), candidate, selected, oriented, orientation,
                        descriptor, best, rejected, error)


def gaussian_pixel_example(data: DeepDiveData, row: int = 50, col: int = 35):
    """Engine kernel and exact REFLECT_101 weighted sum at one source pixel."""
    kernel = scale_space.gaussian_kernel(1.6)
    radius = kernel.shape[0] // 2
    padded = cv2.copyMakeBorder(data.image, radius, radius, radius, radius,
                                 cv2.BORDER_REFLECT_101)
    patch = padded[row:row+2*radius+1, col:col+2*radius+1]
    return patch, kernel, float(np.sum(patch * kernel))


def dog_examples(data: DeepDiveData, coords=((65, 65), (40, 28), (93, 97))):
    octave = data.pyramid[0]
    return tuple((r, c, float(octave.gaussians[1][r, c]),
                  float(octave.gaussians[2][r, c]), float(octave.dogs[1][r, c]))
                 for r, c in coords)


def signed_pixel(data: DeepDiveData, octave: int, layer: int, row: int, col: int):
    values = data.pyramid[octave].dogs[layer]
    return float(values[row, col]), signed_rgba(values)[row, col].copy()
