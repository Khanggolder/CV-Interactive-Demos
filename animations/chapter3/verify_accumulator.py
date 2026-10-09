"""Verify actual Hough votes, the peak, and optionally every review frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hough_accumulator import (
    DISTRACTORS, MAIN_POINTS, RHO_CENTERS, RHO_EDGES, SHAPE, THETA_EDGES,
    THETA_SAMPLES, accumulate, compute_rho, point_votes,
)
from hough_transform import candidate_endpoints, common_line_parameters, parameter_position


ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "media/videos/hough_accumulator/480p15/HoughAccumulatorVoting_review.mp4"


def validate_geometry() -> dict:
    assert np.max(np.abs(MAIN_POINTS[:, 0] + 2 * MAIN_POINTS[:, 1] - 5)) < 1e-12
    assert np.all(np.abs(DISTRACTORS[:, 0] + 2 * DISTRACTORS[:, 1] - 5) > 1)
    points = np.vstack((MAIN_POINTS, DISTRACTORS))
    accumulator = accumulate(points)
    rho_values = np.array([compute_rho(point, THETA_SAMPLES) for point in points])
    # Independent binning route, rather than checking the routine against itself.
    reference, _, _ = np.histogram2d(rho_values.ravel(), np.tile(THETA_SAMPLES, len(points)),
                                    bins=(RHO_EDGES, THETA_EDGES))
    assert np.array_equal(accumulator, reference)
    assert accumulator.sum() == len(points) * len(THETA_SAMPLES) == 162
    winners = np.argwhere(accumulator == accumulator.max())
    assert winners.tolist() == [[14, 6]]
    row, column = map(int, winners[0])
    assert accumulator[row, column] == 6
    assert accumulate(DISTRACTORS)[row, column] == 0
    for point in MAIN_POINTS:
        assert tuple(point_votes(point)[column]) == (row, column)
    rho, theta = float(RHO_CENTERS[row]), float(THETA_SAMPLES[column])
    projection_errors = np.abs(compute_rho(MAIN_POINTS.T, theta) - rho)
    assert projection_errors.max() < 0.25
    foot = rho * np.array([np.cos(theta), np.sin(theta)])
    endpoints = candidate_endpoints(foot, theta)
    assert np.max(np.abs(compute_rho(endpoints.T, theta) - rho)) < 1e-12
    for point in points:
        for angle in np.linspace(0, np.pi, 181):
            assert np.isfinite(candidate_endpoints(point, angle)).all()
            assert np.isfinite(parameter_position(angle, compute_rho(point, angle))).all()
    # Same result under different frame sampling: display is a prefix sum at
    # a given angle, not a counter incremented once per rendered video frame.
    for fps in (15, 30, 60):
        last_count = -1
        for angle in np.linspace(0, np.deg2rad(179.9), fps + 1):
            count = np.searchsorted(THETA_SAMPLES, angle + 1e-10, side="right")
            assert last_count <= count <= 18
            last_count = count
        assert last_count == 18
    true_rho, true_theta = common_line_parameters()
    return {"main_line": "x + 2y = 5", "supporting_points": MAIN_POINTS.tolist(),
            "distractors": DISTRACTORS.tolist(), "shape_rho_theta": list(SHAPE),
            "theta_samples_degrees": np.rad2deg(THETA_SAMPLES).round(8).tolist(),
            "delta_rho": 0.5, "peak_index_zero_based": [row, column],
            "rho_bin": [float(RHO_EDGES[row]), float(RHO_EDGES[row + 1])],
            "theta_bin_degrees": np.rad2deg(THETA_EDGES[column:column + 2]).round(8).tolist(),
            "peak_votes": int(accumulator[row, column]),
            "runner_up_votes": int(np.sort(accumulator.ravel())[-2]), "total_votes": int(accumulator.sum()),
            "detected_rho": rho, "detected_theta_degrees": float(np.rad2deg(theta)),
            "true_rho": true_rho, "true_theta_degrees": float(np.rad2deg(true_theta)),
            "max_support_distance_to_detected_line": float(projection_errors.max()),
            "independent_histogram_match": True, "unique_peak": True, "finite_geometry": True}


def validate_video(contact_sheet=False) -> dict:
    import av
    from PIL import Image, ImageDraw

    targets = iter([1.5, 3.8, 5.8, 8.0, 11.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 30.5])
    target = next(targets)
    samples = []
    count = 0
    with av.open(str(VIDEO)) as container:
        stream = container.streams.video[0]
        width, height, fps = stream.width, stream.height, float(stream.average_rate)
        for frame in container.decode(stream):
            count += 1
            time = float(frame.pts * frame.time_base)
            if contact_sheet and time >= target:
                samples.append((time, frame.to_image()))
                target = next(targets, float("inf"))
    assert (width, height) == (854, 480)
    assert 25 <= count / fps <= 35, count / fps
    if contact_sheet:
        sheet = Image.new("RGB", (width * 3, height * 4), "#0B1020")
        draw = ImageDraw.Draw(sheet)
        for i, (time, frame) in enumerate(samples):
            xy = ((i % 3) * width, (i // 3) * height)
            sheet.paste(frame, xy)
            draw.text((xy[0] + 8, xy[1] + 8), f"{time:.2f}s", fill="yellow")
        sheet.save(ROOT / "media/hough_accumulator/contact_sheet.png")
    return {"path": str(VIDEO), "bytes": VIDEO.stat().st_size, "resolution": [width, height],
            "fps": fps, "frames": count, "duration_seconds": count / fps, "all_frames_decoded": True}


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
