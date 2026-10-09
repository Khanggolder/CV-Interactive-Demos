"""Numerical Hough checks and optional full decoding of the review video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hough_transform import (
    EXTRA_POINTS, IMAGE_BOUNDS, P1, P2, THETA_END, candidate_endpoints,
    common_line_parameters, compute_rho, hough_curve_points, image_position,
    parameter_position,
)


ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "media/videos/hough_transform/480p15/HoughTransformPointToSinusoid_review.mp4"


def validate_geometry() -> dict:
    rho, theta = common_line_parameters()
    points = np.vstack((P1, P2, EXTRA_POINTS))
    assert np.max(np.abs(points[:, 0] + 2 * points[:, 1] - 5)) < 1e-12
    assert abs(rho - np.sqrt(5)) < 1e-12
    assert abs(theta - np.arctan2(2, 1)) < 1e-12
    exact_error = max(abs(float(compute_rho(point, theta)) - rho) for point in points)
    interpolation_errors = []
    line_errors = []
    normal_errors = []
    xmin, xmax, ymin, ymax = IMAGE_BOUNDS
    for point in points:
        samples = hough_curve_points(point)
        assert np.isfinite(samples).all()
        interpolation_errors.append(abs(np.interp(theta, samples[:, 0], samples[:, 1]) - rho))
        for angle in np.linspace(0, THETA_END, 721):
            normal = np.array([np.cos(angle), np.sin(angle)])
            projection = float(compute_rho(point, angle))
            foot = projection * normal
            endpoints = candidate_endpoints(point, angle)
            assert np.isfinite(np.r_[foot, endpoints.ravel()]).all()
            assert np.all(endpoints[:, 0] >= xmin - 1e-10)
            assert np.all(endpoints[:, 0] <= xmax + 1e-10)
            assert np.all(endpoints[:, 1] >= ymin - 1e-10)
            assert np.all(endpoints[:, 1] <= ymax + 1e-10)
            line_errors.append(float(np.max(np.abs(endpoints @ normal - projection))))
            normal_errors.append(abs(float(np.dot(point - foot, normal))))
            assert np.isfinite(parameter_position(angle, projection)).all()
            assert np.isfinite(image_position(foot)).all()
    assert exact_error < 1e-12
    assert max(interpolation_errors) < 1e-4
    assert max(line_errors + normal_errors) < 1e-10
    return {"p1": P1.tolist(), "p2": P2.tolist(), "extra_points": EXTRA_POINTS.tolist(),
            "line": "x + 2y = 5", "rho_star": rho, "theta_star_radians": theta,
            "theta_star_degrees": float(np.rad2deg(theta)),
            "intersection_exact_max_error": exact_error,
            "sampled_curve_intersection_max_error": max(interpolation_errors),
            "sampled_curve_tolerance": 1e-4,
            "candidate_line_max_residual": max(line_errors),
            "normal_perpendicularity_max_error": max(normal_errors),
            "finite_geometry": True}


def validate_video(contact_sheet=False) -> dict:
    import av
    from PIL import Image, ImageDraw

    sample_times = iter([2.7, 5.1, 6.8, 9.5, 11.3, 13.5,
                         15.0, 17.7, 19.8, 22.0, 26.5, 30.8])
    next_sample = next(sample_times)
    samples = []
    count = 0
    with av.open(str(VIDEO)) as container:
        stream = container.streams.video[0]
        width, height = stream.width, stream.height
        fps = float(stream.average_rate)
        for frame in container.decode(stream):
            count += 1
            time = float(frame.pts * frame.time_base)
            if contact_sheet and time >= next_sample:
                samples.append((time, frame.to_image()))
                next_sample = next(sample_times, float("inf"))
    assert (width, height) == (854, 480)
    assert 30 <= count / fps <= 38
    if contact_sheet:
        sheet = Image.new("RGB", (width * 3, height * 4), "#0B1020")
        draw = ImageDraw.Draw(sheet)
        for index, (time, frame) in enumerate(samples):
            xy = ((index % 3) * width, (index // 3) * height)
            sheet.paste(frame, xy)
            draw.text((xy[0] + 8, xy[1] + 8), f"{time:.2f}s", fill="yellow")
        sheet.save(ROOT / "media/hough_transform/contact_sheet.png")
    return {"path": str(VIDEO), "bytes": VIDEO.stat().st_size,
            "resolution": [width, height], "fps": fps, "frames": count,
            "duration_seconds": count / fps, "all_frames_decoded": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--contact-sheet", action="store_true")
    args = parser.parse_args()
    report = {"geometry": validate_geometry()}
    if args.video or args.contact_sheet:
        report["video"] = validate_video(args.contact_sheet)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
