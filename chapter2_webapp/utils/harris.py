"""Transparent Harris/Shi-Tomasi pipeline on small grayscale images.

Coordinates are (row, col), displayed as (x=col, y=row) with y down.
Sobel uses correlation kernels Kx=[[-1,0,1],[-2,0,2],[-1,0,1]]/8
and Ky=Kx.T/8, cv2.BORDER_REFLECT_101. The /8 factor gives unit
derivative for an interior unit ramp. Tensor integration is a normalized
odd box or finite Gaussian window; it is distinct from differentiation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8
KY = KX.T.copy()
PRESET_NAMES = (
    "Vùng phẳng", "Cạnh ngang", "Cạnh dọc", "Góc chữ L",
    "Bàn cờ", "Góc có nhiễu", "Góc xoay", "Góc đổi tỉ lệ",
)


def presets(size: int = 41) -> dict[str, np.ndarray]:
    if size < 21 or size % 2 != 1:
        raise ValueError("size must be odd and >= 21")
    yy, xx = np.indices((size, size))
    middle = size // 2
    corner = np.where((xx >= middle) & (yy >= middle), 1.0, 0.0)
    rng = np.random.default_rng(2026)
    rotation = cv2.warpAffine(corner, cv2.getRotationMatrix2D((middle, middle), 28, 1),
                              (size, size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    scaled = cv2.warpAffine(corner, cv2.getRotationMatrix2D((middle, middle), 0, 0.6),
                            (size, size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return {
        PRESET_NAMES[0]: np.full((size, size), 0.35),
        PRESET_NAMES[1]: (yy >= middle).astype(np.float64),
        PRESET_NAMES[2]: (xx >= middle).astype(np.float64),
        PRESET_NAMES[3]: corner,
        PRESET_NAMES[4]: (((xx // 7) + (yy // 7)) % 2).astype(np.float64),
        PRESET_NAMES[5]: np.clip(corner + rng.normal(0, 0.13, corner.shape), 0, 1),
        PRESET_NAMES[6]: rotation,
        PRESET_NAMES[7]: scaled,
    }


def _image(image: np.ndarray) -> np.ndarray:
    result = np.asarray(image, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 7 or not np.all(np.isfinite(result)):
        raise ValueError("Expected a finite 2D image at least 7x7")
    return result


def _window(image: np.ndarray, row: int, col: int, size: int) -> np.ndarray:
    if size < 3 or size % 2 != 1:
        raise ValueError("Window must be odd and >= 3")
    radius = size // 2
    if row - radius < 0 or col - radius < 0 or row + radius >= image.shape[0] or col + radius >= image.shape[1]:
        raise ValueError("Window would cross image boundary")
    return image[row-radius:row+radius+1, col-radius:col+radius+1]


def shifted_patches(image: np.ndarray, row: int, col: int, size: int,
                    u: int, v: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact comparison, no padding or wraparound. u is x/column, v is y/row."""
    image = _image(image)
    return _window(image, row, col, size), _window(image, row+v, col+u, size)


def shift_energy(image: np.ndarray, row: int, col: int, size: int,
                 u: int, v: int) -> float:
    before, after = shifted_patches(image, row, col, size, u, v)
    return float(np.sum((after-before)**2))


def energy_grid(image: np.ndarray, row: int, col: int, size: int,
                radius: int = 3) -> np.ndarray:
    return np.array([[shift_energy(image, row, col, size, u, v)
                      for u in range(-radius, radius+1)]
                     for v in range(-radius, radius+1)])


def gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = _image(image)
    return (cv2.filter2D(image, cv2.CV_64F, KX, borderType=cv2.BORDER_REFLECT_101),
            cv2.filter2D(image, cv2.CV_64F, KY, borderType=cv2.BORDER_REFLECT_101))


def taylor_energy(image: np.ndarray, row: int, col: int, size: int,
                  u: float, v: float) -> float:
    ix, iy = gradients(image)
    return float(np.sum((u * _window(ix, row, col, size) +
                         v * _window(iy, row, col, size))**2))


def tensor_maps(image: np.ndarray, size: int = 5, sigma: float = 1.2,
                weighting: str = "Gaussian") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = _image(image)
    ix, iy = gradients(image)
    kernel = integration_kernel(size, sigma, weighting)
    smooth = lambda a: cv2.filter2D(a, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT_101)
    return smooth(ix*ix), smooth(ix*iy), smooth(iy*iy)


def integration_kernel(size: int = 5, sigma: float = 1.2,
                       weighting: str = "Gaussian") -> np.ndarray:
    """Normalized finite support used for the tensor, not for differentiation."""
    if size < 3 or size % 2 != 1 or sigma <= 0:
        raise ValueError("Invalid integration window or sigma")
    if weighting == "Gaussian":
        g = cv2.getGaussianKernel(size, sigma, cv2.CV_64F)
        kernel = g @ g.T
    elif weighting == "Box":
        kernel = np.full((size, size), 1 / (size * size))
    else:
        raise ValueError("weighting must be Gaussian or Box")
    return kernel


def tensor_at(maps: tuple[np.ndarray, np.ndarray, np.ndarray],
              row: int, col: int) -> np.ndarray:
    a, b, c = (float(m[row, col]) for m in maps)
    return np.array([[a, b], [b, c]], dtype=np.float64)


def eigenvalues(maps: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = maps
    delta = np.sqrt(np.maximum(0, (a-c)**2 + 4*b*b))
    return np.maximum(0, (a+c+delta)/2), np.maximum(0, (a+c-delta)/2)


def harris_response(maps: tuple[np.ndarray, np.ndarray, np.ndarray], k: float = 0.05) -> np.ndarray:
    if not 0 < k < 0.25:
        raise ValueError("k must be between 0 and 0.25")
    a, b, c = maps
    return a*c - b*b - k*(a+c)**2


def shi_tomasi_response(maps: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return eigenvalues(maps)[1]


def classify_eigenvalues(large: float, small: float, threshold: float = 0.005) -> str:
    if large < threshold:
        return "Vùng phẳng"
    if small < threshold:
        return "Cạnh"
    return "Góc"


def threshold_nms(response: np.ndarray, threshold: float, size: int = 5) -> np.ndarray:
    """Absolute threshold; excludes radius-wide border; equal local peaks: top-left wins."""
    response = _image(response)
    if size < 3 or size % 2 != 1 or threshold < 0:
        raise ValueError("Invalid NMS size or threshold")
    radius = size // 2
    result = np.zeros(response.shape, dtype=bool)
    for row in range(radius, response.shape[0]-radius):
        for col in range(radius, response.shape[1]-radius):
            if _nms_decision(response, threshold, radius, row, col)[0]:
                result[row, col] = True
    return result


def _nms_decision(response: np.ndarray, threshold: float, radius: int,
                  row: int, col: int) -> tuple[bool, str]:
    if row < radius or col < radius or row >= response.shape[0]-radius or col >= response.shape[1]-radius:
        return False, "REJECT · nằm trong dải biên NMS"
    value = float(response[row, col])
    if value <= threshold:
        return False, "REJECT · không vượt ngưỡng"
    neighborhood = response[row-radius:row+radius+1, col-radius:col+radius+1]
    if value < float(np.max(neighborhood)):
        return False, "REJECT · có điểm lớn hơn trong lân cận"
    if tuple(np.argwhere(neighborhood == value)[0]) != (radius, radius):
        return False, "REJECT · đồng hạng nhưng tọa độ (hàng,cột) lớn hơn"
    return True, "KEEP · vượt ngưỡng và là cực đại cục bộ đầu tiên"


def nms_decision(response: np.ndarray, threshold: float, size: int,
                 row: int, col: int) -> tuple[bool, str]:
    """Explain an individual real NMS decision using the same rule as the mask."""
    response = _image(response)
    if size < 3 or size % 2 != 1 or threshold < 0 or not 0 <= row < response.shape[0] or not 0 <= col < response.shape[1]:
        raise ValueError("Invalid NMS decision arguments")
    return _nms_decision(response, threshold, size // 2, row, col)


@dataclass(frozen=True)
class Pipeline:
    ix: np.ndarray
    iy: np.ndarray
    maps: tuple[np.ndarray, np.ndarray, np.ndarray]
    large: np.ndarray
    small: np.ndarray
    harris: np.ndarray
    shi_tomasi: np.ndarray


def pipeline(image: np.ndarray, size: int = 5, sigma: float = 1.2,
             weighting: str = "Gaussian", k: float = 0.05) -> Pipeline:
    ix, iy = gradients(image)
    maps = tensor_maps(image, size, sigma, weighting)
    large, small = eigenvalues(maps)
    return Pipeline(ix, iy, maps, large, small,
                    harris_response(maps, k), small)
