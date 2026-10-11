"""Pure numerical LSD teaching prototype (not the published reference detector).

Arrays use [y, x], with Cartesian y increasing upward. Region growing compares
axial orientations, while NFA validation retains directed gradient polarity.
"""

from __future__ import annotations

from collections import deque
from math import exp, fsum, lgamma, log, log1p, pi

import numpy as np

DATA_SEED = 9
SHAPE = (10, 12)
DEFAULT_TAU = pi / 8
DEFAULT_MIN_MAGNITUDE = .09
DEFAULT_EPSILON = 1.0
MAX_PIXELS = 1600

PRESETS = (
    "A · Approved reference", "B · Clean strong line", "C · Noisy texture",
    "D · Short fragments", "E · Crossing structures", "F · Contrast reversal",
)

PRESET_NOTES = {
    PRESETS[0]: ("Pixel gradient nào mạnh nhất?", "Một cấu trúc mạnh và một vùng ngắn cho NFA khác nhau.",
                 "Đây là score của mô hình giảng dạy, không phải bảo đảm biên vật thể."),
    PRESETS[1]: ("Biên dài, sạch có tạo LSR nhất quán?", "Theo dõi n, k và NFA được tính lại.",
                 "Một biên mạnh vẫn có thể bị chia nếu τ quá hẹp."),
    PRESETS[2]: ("Gradient mạnh do texture có đủ để ACCEPT?", "So sánh orientation và NFA thật.",
                 "Nhiễu có thể tình cờ tạo support hợp lệ."),
    PRESETS[3]: ("Vùng support ngắn tạo bằng chứng ra sao?", "Quan sát n, k của từng candidate.",
                 "Đoạn ngắn không mặc nhiên sai; kết luận tùy score và ε."),
    PRESETS[4]: ("Hai hướng cắt nhau sẽ nối hay tách?", "Xem BFS và candidate tại giao điểm.",
                 "Một seed cố định có thể bỏ qua nhánh khác hướng."),
    PRESETS[5]: ("Đảo toàn bộ polarity có đổi n/k không?", "So với A: tangent fitted cũng đảo nên n/k có thể giữ nguyên.",
                 "Chỉ pha trộn polarity trong một LSR mới có thể làm k directed giảm dù axial growth giữ vùng."),
}


def validate_intensity_field(intensity):
    array = np.asarray(intensity, dtype=float)
    if array.ndim != 2 or not 4 <= array.shape[0] <= 40 or not 4 <= array.shape[1] <= 40:
        raise ValueError("Intensity phải là ma trận 2D, mỗi chiều 4–40")
    if array.size > MAX_PIXELS or not np.isfinite(array).all():
        raise ValueError("Intensity phải hữu hạn và không quá 1600 pixel")
    return array.copy()


def validate_parameters(tau=DEFAULT_TAU, min_magnitude=DEFAULT_MIN_MAGNITUDE,
                        epsilon=DEFAULT_EPSILON):
    values = (float(tau), float(min_magnitude), float(epsilon))
    if not all(np.isfinite(v) for v in values) or not 0 < values[0] < pi / 2 or values[1] < 0 or values[2] <= 0:
        raise ValueError("Cần 0<τ<90°, minimum magnitude≥0 và ε>0 hữu hạn")
    return values


def generate_lsd_reference_field():
    y, x = np.indices(SHAPE, dtype=float)
    edge = .5 + .42 * np.tanh((y - .55*x - 1.3) / .8)
    blob = .40 * np.exp(-((x-8.8)**2 + (y-1.6)**2) / (2*.75**2))
    return edge + blob + np.random.default_rng(DATA_SEED).normal(0, .055, SHAPE)


def preset_field(name):
    if name not in PRESETS:
        raise ValueError("Preset không hợp lệ")
    if name == PRESETS[0]:
        return generate_lsd_reference_field()
    if name == PRESETS[5]:
        return 1 - generate_lsd_reference_field()
    y, x = np.indices((12, 14), dtype=float)
    if name == PRESETS[1]:
        return .5 + .43 * np.tanh((y - .52*x - 1.4) / .7)
    if name == PRESETS[2]:
        base = .5 + .26 * np.tanh((y - .52*x - 1.4) / .8)
        texture = .16 * np.sin(2.1*x + 1.7*y)
        return base + texture + np.random.default_rng(37).normal(0, .13, (12, 14))
    if name == PRESETS[3]:
        return .46 + .40 * np.exp(-((x-3)**2 + (y-3)**2) / 2) + .34 * np.exp(-((x-10)**2 + (y-8)**2) / 2)
    return .5 + .24*np.tanh((y-5.5)/.6) + .24*np.tanh((x-6.5)/.6)


def compute_gradient_field(intensity):
    gy, gx = np.gradient(validate_intensity_field(intensity))
    return gx, gy, np.hypot(gx, gy)


def compute_level_line_angles(gx, gy):
    gx, gy = np.asarray(gx, dtype=float), np.asarray(gy, dtype=float)
    if gx.ndim != 2 or gx.shape != gy.shape or not np.isfinite(gx).all() or not np.isfinite(gy).all():
        raise ValueError("Gx, Gy phải cùng shape và hữu hạn")
    return (np.arctan2(gy, gx) - pi/2) % (2*pi)


def angular_difference_pi(a, b):
    return np.abs((np.asarray(a) - b + pi/2) % pi - pi/2)


def angular_difference_2pi(a, b):
    return np.abs((np.asarray(a) - b + pi) % (2*pi) - pi)


def neighbors(pixel, shape):
    y, x = map(int, pixel)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy or dx) and 0 <= y+dy < shape[0] and 0 <= x+dx < shape[1]:
                yield y+dy, x+dx


def select_lsd_seed(magnitude, used, min_magnitude=DEFAULT_MIN_MAGNITUDE):
    magnitude, used = np.asarray(magnitude, dtype=float), np.asarray(used, dtype=bool)
    min_magnitude = float(min_magnitude)
    if (magnitude.shape != used.shape or magnitude.ndim != 2 or
            not np.isfinite(magnitude).all() or np.any(magnitude < 0) or
            not np.isfinite(min_magnitude) or min_magnitude < 0):
        raise ValueError("Magnitude/used phải cùng shape 2D")
    eligible = (~used) & (magnitude > 0) & (magnitude >= min_magnitude)
    if not eligible.any():
        return None
    return tuple(map(int, np.unravel_index(np.argmax(np.where(eligible, magnitude, -np.inf)), magnitude.shape)))


def grow_lsr(seed, angles, magnitude, used, tau=DEFAULT_TAU,
             min_magnitude=DEFAULT_MIN_MAGNITUDE):
    """BFS matching the Manim seed, neighbor order, used and visited semantics."""
    tau, min_magnitude, _ = validate_parameters(tau, min_magnitude)
    angles, magnitude = np.asarray(angles, dtype=float), np.asarray(magnitude, dtype=float)
    if (not isinstance(used, np.ndarray) or used.dtype != bool or angles.ndim != 2 or
            angles.shape != magnitude.shape or used.shape != magnitude.shape or
            not np.isfinite(angles).all() or not np.isfinite(magnitude).all() or np.any(magnitude < 0)):
        raise ValueError("Fields/used phải cùng shape")
    seed = tuple(map(int, seed))
    if (not 0 <= seed[0] < magnitude.shape[0] or not 0 <= seed[1] < magnitude.shape[1]
            or used[seed] or magnitude[seed] <= 0 or magnitude[seed] < min_magnitude):
        raise ValueError("Seed không hợp lệ")
    reference = float(angles[seed])
    queue, visited, members = deque([seed]), {seed}, [seed]
    used[seed] = True
    events = []
    while queue:
        parent = queue.popleft()
        for point in neighbors(parent, magnitude.shape):
            if point in visited or used[point] or magnitude[point] <= 0 or magnitude[point] < min_magnitude:
                continue
            visited.add(point)
            difference = float(angular_difference_pi(angles[point], reference))
            accepted = difference < tau
            events.append({"parent": parent, "pixel": point, "magnitude": float(magnitude[point]),
                           "angle": float(angles[point]), "reference": reference,
                           "difference": difference, "tau": tau, "accepted": bool(accepted),
                           "reason": "aligned" if accepted else "axial angle ≥ τ"})
            if accepted:
                used[point] = True
                queue.append(point)
                members.append(point)
    return {"seed": seed, "reference": reference, "pixels": members, "events": events}


def replay_lsr_events(region, count):
    events = region["events"]
    if not 0 <= count <= len(events):
        raise ValueError("Replay prefix ngoài phạm vi")
    members, rejected, processed = [region["seed"]], [], []
    for event in events[:count]:
        processed.append(event["pixel"])
        (members if event["accepted"] else rejected).append(event["pixel"])
    return {"members": members, "rejected": rejected, "processed": processed,
            "current": events[count-1] if count else None, "count": count}


def fit_lsr_segment(region, magnitude, angles, tau=DEFAULT_TAU):
    tau, _, _ = validate_parameters(tau, 0)
    magnitude, angles = np.asarray(magnitude, dtype=float), np.asarray(angles, dtype=float)
    if (magnitude.ndim != 2 or magnitude.shape != angles.shape or
            not np.isfinite(magnitude).all() or not np.isfinite(angles).all()):
        raise ValueError("Magnitude/angles phải cùng shape 2D hữu hạn")
    raw_pixels = np.asarray(region["pixels"], dtype=float)
    if (raw_pixels.ndim != 2 or raw_pixels.shape[1] != 2 or not len(raw_pixels) or
            not np.isfinite(raw_pixels).all() or not np.array_equal(raw_pixels, np.floor(raw_pixels)) or
            np.any(raw_pixels < 0) or np.any(raw_pixels[:, 0] >= magnitude.shape[0]) or
            np.any(raw_pixels[:, 1] >= magnitude.shape[1]) or
            not np.isfinite(region["reference"])):
        raise ValueError("Region rỗng hoặc pixel không hợp lệ")
    pixels = raw_pixels.astype(int)
    if len(set(map(tuple, pixels))) != len(pixels):
        raise ValueError("Region chứa pixel trùng")
    xy = pixels[:, ::-1].astype(float)
    weights = np.asarray(magnitude)[tuple(pixels.T)]
    if np.any(weights < 0) or not np.isfinite(weights).all() or fsum(weights) <= 0:
        raise ValueError("Trọng số gradient không hợp lệ")
    center = np.average(xy, axis=0, weights=weights)
    covariance = ((xy-center)*weights[:, None]).T @ (xy-center)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if len(xy) == 1 or eigenvalues[-1] <= 1e-14:
        tangent = np.array([np.cos(region["reference"]), np.sin(region["reference"])])
    else:
        tangent = eigenvectors[:, -1]
    member_angles = np.asarray(angles)[tuple(pixels.T)]
    direction_sum = np.sum(weights[:, None] * np.column_stack((np.cos(member_angles), np.sin(member_angles))), axis=0)
    if tangent @ direction_sum < 0:
        tangent *= -1
    normal = np.array([-tangent[1], tangent[0]])
    along, across = (xy-center) @ tangent, (xy-center) @ normal
    pad_t, pad_n = .5*np.abs(tangent).sum(), .5*np.abs(normal).sum()
    amin, amax = along.min()-pad_t, along.max()+pad_t
    bmin, bmax = across.min()-pad_n, across.max()+pad_n
    rectangle = np.array([center+a*tangent+b*normal for a, b in
                          ((amin, bmin), (amax, bmin), (amax, bmax), (amin, bmax))])
    endpoints = np.array([center+along.min()*tangent, center+along.max()*tangent])
    angle = float(np.arctan2(tangent[1], tangent[0]) % (2*pi))
    aligned = angular_difference_2pi(member_angles, angle) < tau
    return {"center": center, "tangent": tangent, "normal": normal, "endpoints": endpoints,
            "rectangle": rectangle, "length": float(amax-amin), "width": float(bmax-bmin),
            "angle": angle, "aligned": aligned, "n": len(xy), "k": int(aligned.sum())}


def log_binomial_tail(n, k, p):
    if not isinstance(n, (int, np.integer)) or not isinstance(k, (int, np.integer)) or not 0 <= k <= n or n < 0:
        raise ValueError("Cần số nguyên 0≤k≤n")
    p = float(p)
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Cần 0≤p≤1")
    if k == 0 or p == 1:
        return 0.0
    if p == 0:
        return -float("inf")
    if k == n:
        return n*log(p)
    terms = [lgamma(n+1)-lgamma(j+1)-lgamma(n-j+1)+j*log(p)+(n-j)*log1p(-p)
             for j in range(k, n+1)]
    maximum = max(terms)
    return min(0.0, maximum + log(fsum(exp(term-maximum) for term in terms)))


def compute_binomial_tail(n, k, p):
    return exp(log_binomial_tail(n, k, p))


def compute_nfa(n, k, tests, p0=DEFAULT_TAU/pi):
    if not isinstance(tests, (int, np.integer)) or tests < 1:
        raise ValueError("N_T phải là số nguyên dương")
    return exp(log(tests) + log_binomial_tail(n, k, p0))


def format_probability(log_value):
    if log_value == -float("inf"):
        return "0 (xác suất toán học đúng bằng 0)"
    if log_value < log(1e-300):
        exponent = int(np.floor(log_value/log(10)))
        mantissa = exp(log_value-exponent*log(10))
        return f"{mantissa:.3f} × 10^{exponent}"
    return f"{exp(log_value):.6g}"


def analyze_field(intensity, tau=DEFAULT_TAU, min_magnitude=DEFAULT_MIN_MAGNITUDE,
                  epsilon=DEFAULT_EPSILON):
    tau, min_magnitude, epsilon = validate_parameters(tau, min_magnitude, epsilon)
    intensity = validate_intensity_field(intensity)
    gx, gy, magnitude = compute_gradient_field(intensity)
    angles = compute_level_line_angles(gx, gy)
    used = np.zeros(intensity.shape, dtype=bool)
    regions = []
    while (seed := select_lsd_seed(magnitude, used, min_magnitude)) is not None:
        region = grow_lsr(seed, angles, magnitude, used, tau, min_magnitude)
        region["fit"] = fit_lsr_segment(region, magnitude, angles, tau)
        regions.append(region)
    tests = len(regions)
    for region in regions:
        fit = region["fit"]
        log_tail = log_binomial_tail(fit["n"], fit["k"], tau/pi)
        region["log_tail"] = log_tail
        region["log_nfa"] = log(tests) + log_tail
        region["nfa"] = exp(region["log_nfa"])
        region["accept"] = region["log_nfa"] < log(epsilon)
    return {"intensity": intensity, "gx": gx, "gy": gy, "magnitude": magnitude,
            "angles": angles, "regions": regions, "tests": tests, "tau": tau,
            "min_magnitude": min_magnitude, "epsilon": epsilon}
