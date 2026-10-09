"""Independent circle/gradient audit, frame-bound preflight and full decode."""

import argparse
import json

import numpy as np

from hough_circle import (
    CENTER, RADIUS, ROOT, IMAGE_UNIT, CENTER_UNIT, EDGE_WIDTH, LAYER_RADII,
    HoughCircleTransform, candidate_center, center_locus_for_point, center_position,
    generate_circle_example, image_position, layer_position, synthetic_intensity,
)

OUTPUT = ROOT / "media/hough_circle"
VIDEO = ROOT / "media/videos/hough_circle/480p15/HoughCircleTransform_review.mp4"


def validate():
    data = generate_circle_example()
    points, gradients = data["points"], data["gradients"]
    distances = np.linalg.norm(points - CENTER, axis=1)
    residuals = np.sum((points - CENTER)**2, axis=1) - RADIUS**2
    np.testing.assert_allclose(distances, RADIUS, atol=1e-12, rtol=0)
    np.testing.assert_allclose(residuals, 0, atol=1e-12, rtol=0)
    # Subtract equal-radius circle equations to solve for the intersection,
    # independently of both the supplied center and the gradient votes.
    matrix = 2*(points[1:] - points[0])
    rhs = np.sum(points[1:]**2, axis=1) - np.sum(points[0]**2)
    solved, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    assert rank == 2
    intersection_error = float(np.linalg.norm(solved - CENTER))
    assert intersection_error < 1e-12
    third_point_solution = np.linalg.solve(matrix[:2], rhs[:2])
    np.testing.assert_allclose(third_point_solution, CENTER, atol=1e-12)
    unit_gradients = gradients / np.linalg.norm(gradients, axis=1)[:, None]
    np.testing.assert_allclose(unit_gradients, (CENTER-points)/RADIUS, atol=1e-12)
    vote_errors = np.linalg.norm(data["gradient_votes"] - CENTER, axis=1)
    assert vote_errors.max() < 1e-12
    h = 1e-6
    numerical_gradient = np.column_stack([
        (synthetic_intensity(points + h*axis) - synthetic_intensity(points - h*axis))/(2*h)
        for axis in np.eye(2)])
    np.testing.assert_allclose(gradients, numerical_gradient, atol=2e-8, rtol=2e-8)
    records = []
    for i, point in enumerate(points):
        locus = center_locus_for_point(point)
        np.testing.assert_allclose(np.linalg.norm(locus-point, axis=1), RADIUS, atol=1e-12)
        for r in LAYER_RADII:
            layer = center_locus_for_point(point, r)
            np.testing.assert_allclose(np.linalg.norm(layer-point, axis=1), r, atol=1e-12)
            if i == 0:
                projected = np.array([layer_position(ab, r, point) for ab in layer])
                assert np.isfinite(projected).all()
                assert (np.abs(projected[:, :2]) < [7.9, 3.0]).all()
        records.append(dict(point=point.tolist(), source_angle_degrees=int(data["angles_degrees"][i]),
                            distance_to_true_center=float(distances[i]),
                            fixed_locus=dict(center=point.tolist(), radius=RADIUS,
                                             residual_at_true_center=float(residuals[i])),
                            gradient=gradients[i].tolist(), unit_gradient=unit_gradients[i].tolist(),
                            gradient_theta_degrees=float(np.degrees(np.arctan2(gradients[i, 1], gradients[i, 0]))),
                            gradient_predicted_center=data["gradient_votes"][i].tolist(),
                            gradient_center_error=float(vote_errors[i])))
    # Audit the whole compass sweep, including the full candidate circle.
    for phi in np.linspace(-np.pi/2, 3*np.pi/2, 361):
        candidate = candidate_center(points[0], RADIUS, phi)
        assert abs(np.linalg.norm(candidate - points[0]) - RADIUS) < 1e-12
        screen = image_position(candidate)
        assert screen[0] - IMAGE_UNIT*RADIUS > -7.9
        assert screen[0] + IMAGE_UNIT*RADIUS < -0.5
        assert abs(screen[1]) + IMAGE_UNIT*RADIUS < 3.0
    for array in (points, gradients, data["gradient_votes"], distances, residuals):
        assert np.isfinite(array).all()
    assert all(np.isfinite(center_position(p)).all() for p in points)
    return dict(true_center=CENTER.tolist(), radius=RADIUS, edge_points=records,
                solved_common_center=solved.tolist(), locus_intersection_error=intersection_error,
                max_locus_residual=float(np.max(np.abs(residuals))),
                max_gradient_center_error=float(vote_errors.max()),
                gradient_polarity="Bright inside, dark outside: inward gradient; center = p + r * unit_gradient.",
                intensity_model=f"I=0.5+0.5*tanh((r0-distance_to_center)/{EDGE_WIDTH})",
                gradient_finite_difference_check=True, radius_layers=LAYER_RADII.tolist(),
                visual_scaling=dict(image_units=IMAGE_UNIT, center_units=CENTER_UNIT,
                                    layer_projection="(0.70*da+0.20*db, 0.22*db+0.95*r)",
                                    image_gradient_arrow_length=1.5, sampled_candidate_dots=24),
                coordinates="Cartesian y up in both spaces", checks_passed=True,
                limitations=["Exact continuous loci; no discretized 3D accumulator or noisy-circle detector.",
                             "24 candidate dots illustrate alternative centers, not accumulated vote counts.",
                             "Known inward polarity gives one candidate per radius; unknown polarity requires +/-."])


def check_video(qa=False):
    import av
    from PIL import Image, ImageDraw
    targets = iter([2, 5.5, 9, 12, 15, 18, 21, 23.5, 26, 29, 31, 33])
    target = next(targets)
    samples = []
    with av.open(str(VIDEO)) as container:
        stream = container.streams.video[0]
        width, height, fps = stream.width, stream.height, float(stream.average_rate)
        count, last_time = 0, -1
        for frame in container.decode(stream):
            seconds = float(frame.pts * frame.time_base)
            assert seconds > last_time
            last_time = seconds
            count += 1
            if qa and seconds >= target:
                samples.append((seconds, frame.to_image()))
                target = next(targets, float("inf"))
        assert count == stream.frames
    assert (width, height, fps) == (854, 480, 15)
    assert 30 <= count/fps <= 38, count/fps
    if qa:
        sheet = Image.new("RGB", (width*3, height*4), "#0B1020")
        draw = ImageDraw.Draw(sheet)
        for i, (seconds, frame) in enumerate(samples):
            x, y = i%3 * width, i//3 * height
            sheet.paste(frame, (x, y))
            draw.text((x+8, y+8), f"{seconds:.2f}s", fill="yellow")
            if i in (4, 7, 9):
                frame.save(OUTPUT / f"review_{seconds:.2f}s.png")
        sheet.save(OUTPUT / "contact_sheet.png")
    return dict(path=str(VIDEO), bytes=VIDEO.stat().st_size, duration=count/fps,
                fps=fps, width=width, height=height, frames=count, all_frames_decoded=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--qa", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT / "validation.json"
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    report["numerical"] = validate()
    if args.preflight:
        from manim import tempconfig
        with tempconfig(dict(dry_run=True, skip_animations=True, write_to_movie=False,
                             save_last_frame=False, pixel_width=854, pixel_height=480,
                             frame_rate=15, progress_bar="none")):
            scene = HoughCircleTransform()
            # Manim's standard 480p width is 854 (rounded to an even pixel),
            # so permit the sub-one-pixel aspect rounding of a 16:9 target.
            assert abs(scene.camera.frame_width / scene.camera.frame_height - 16/9) < 1/480
            scene.render()
        report["layout_preflight"] = "PASS: 16:9 target (854x480 raster rounding), finite geometry, frame bounds, no leftover updaters."
    if args.video or args.qa:
        report["video"] = check_video(args.qa)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"numerical": "PASS", "intersection_error": report["numerical"]["locus_intersection_error"],
                      "gradient_error": report["numerical"]["max_gradient_center_error"],
                      "layout": report.get("layout_preflight"), "video": report.get("video")}, indent=2))
