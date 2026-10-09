"""Independent numerical checks and complete decode of the RANSAC review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ransac import (
    BOUNDS, DATA_SEED, EPSILON, SAMPLE_SEED, TRUE_INTERCEPT, TRUE_SLOPE, UNIT,
    band_vertices, classify_inliers, experiment, fit_line_to_consensus,
    least_squares, line_endpoints, line_from_two_points, normalize_line,
    point_line_distance, position,
)

ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "media/videos/ransac/480p15/RANSACLineFitting_review.mp4"


def slope_intercept(model):
    return (-model[[0, 2]] / model[1]).tolist()


def validate_numeric():
    points, intended, pairs, models, masks, best, final = experiment()
    assert np.array_equal(points, experiment()[0])
    assert len(points) == 20 and intended.sum() == 14
    truth = normalize_line([-TRUE_SLOPE, 1, -TRUE_INTERCEPT])
    assert np.array_equal(classify_inliers(points, truth), intended)
    counts = [int(mask.sum()) for mask in masks]
    assert counts == [3, 7, 14] and best == 2
    assert np.array_equal(masks[best], intended)
    trials = []
    for pair, model, mask in zip(pairs, models, masks):
        p, q = points[pair]
        delta = q - p
        assert np.linalg.norm(delta) > 1e-6
        # Independent cross-product distance, not a second call to classifier.
        cross = delta[0] * (points[:, 1] - p[1]) - delta[1] * (points[:, 0] - p[0])
        independent = np.abs(cross) / np.linalg.norm(delta)
        assert np.allclose(independent, point_line_distance(points, model))
        assert np.array_equal(independent <= EPSILON, mask)
        assert point_line_distance(points[pair], model).max() < 1e-12
        trials.append({"sample_indices_zero_based": pair.tolist(),
                       "sample_coordinates": points[pair].tolist(),
                       "normalized_abc": model.tolist(), "consensus": int(mask.sum()),
                       "inlier_indices": np.flatnonzero(mask).tolist()})
    assert not intended[pairs[0]].all() and intended[pairs[best]].all()
    selected = points[masks[best]]
    center = selected.mean(axis=0)
    covariance = np.cov(selected.T)
    _, eigenvectors = np.linalg.eigh(covariance)
    independent_final = normalize_line([*eigenvectors[:, 0], -center @ eigenvectors[:, 0]])
    assert np.allclose(final, independent_final)
    assert np.array_equal(classify_inliers(points, final), intended)
    residual = point_line_distance(selected, final)
    before = point_line_distance(selected, models[best])
    assert (residual ** 2).sum() < (before ** 2).sum()
    slope, intercept = slope_intercept(final)
    assert abs(slope - TRUE_SLOPE) < 0.03 and abs(intercept - TRUE_INTERCEPT) < 0.06
    xmin, xmax, ymin, ymax = BOUNDS
    assert np.all((points[:, 0] > xmin) & (points[:, 0] < xmax))
    assert np.all((points[:, 1] > ymin) & (points[:, 1] < ymax))
    # Include vertical/horizontal lines in geometry tests, even though none of
    # the displayed candidates require a special-case slope representation.
    geometries = models + [final, least_squares(points), least_squares(points[:18]),
                          normalize_line([1, 0, -0.5]), normalize_line([0, 1, 0])]
    for model in geometries:
        ends, vertices = line_endpoints(model), band_vertices(model)
        assert np.isfinite(ends).all() and np.isfinite(vertices).all()
        assert point_line_distance(ends, model).max() < 1e-12
        assert point_line_distance(vertices, model).max() <= EPSILON + 1e-12
        for v in vertices:
            on_plot = any(np.isclose(v[axis], bound) for axis, bound in
                          ((0, xmin), (0, xmax), (1, ymin), (1, ymax)))
            assert on_plot or np.isclose(point_line_distance(v, model), EPSILON)
        signed = vertices @ model[:2] + model[2]
        assert np.isclose(signed.min(), -EPSILON) and np.isclose(signed.max(), EPSILON)
        # A point exactly epsilon from the line is UNIT*epsilon away onscreen.
        foot = -model[2] * model[:2]
        assert np.isclose(np.linalg.norm(position(foot + EPSILON * model[:2]) - position(foot)),
                          UNIT * EPSILON)
        for p in np.vstack((points, ends, vertices)):
            rendered = position(p)
            assert np.isfinite(rendered).all()
            assert abs(rendered[0]) < 7.8 and abs(rendered[1]) < 4.3
    try:
        line_from_two_points(points[0], points[0])
    except ValueError:
        pass
    else:
        raise AssertionError("Degenerate pair must be rejected")
    return {"data_seed": DATA_SEED, "sample_seed": SAMPLE_SEED,
            "sampling_note": "First three draws in order; seeds chosen for teaching, not a convergence guarantee",
            "true_line": "y = 0.65x + 0.3", "dataset": points.tolist(),
            "total": len(points), "intended_inliers": int(intended.sum()), "intended_outliers": 6,
            "epsilon_perpendicular": EPSILON, "trials": trials, "best_trial_one_based": best + 1,
            "final_normalized_abc": final.tolist(), "final_slope_intercept": [slope, intercept],
            "slope_intercept_error": [slope - TRUE_SLOPE, intercept - TRUE_INTERCEPT],
            "final_mean_perpendicular_residual": float(residual.mean()),
            "final_rms_perpendicular_residual": float(np.sqrt(np.mean(residual ** 2))),
            "candidate_rms_perpendicular_residual": float(np.sqrt(np.mean(before ** 2))),
            "ls_all_slope_intercept": slope_intercept(least_squares(points)),
            "independent_distances_match": True, "independent_tls_match": True,
            "exact_epsilon_band": True, "finite_geometry_and_point_bounds": True}


def validate_video(contact_sheet=False):
    import av
    from PIL import Image, ImageDraw

    samples = []
    targets = iter([1.5, 4, 7.5, 10, 12.5, 15.5, 19, 22, 25, 28, 31, 33])
    target = next(targets)
    with av.open(str(VIDEO)) as container:
        stream = container.streams.video[0]
        width, height, fps = stream.width, stream.height, float(stream.average_rate)
        count = 0
        for frame in container.decode(stream):
            count += 1
            time = float(frame.pts * frame.time_base)
            if contact_sheet and time >= target:
                samples.append((time, frame.to_image()))
                target = next(targets, float("inf"))
    assert (width, height, fps) == (854, 480, 15)
    assert 30 <= count / fps <= 40, count / fps
    if contact_sheet:
        sheet = Image.new("RGB", (width * 3, height * 4), "#0B1020")
        draw = ImageDraw.Draw(sheet)
        for i, (time, frame) in enumerate(samples):
            xy = (i % 3 * width, i // 3 * height)
            sheet.paste(frame, xy)
            draw.text((xy[0] + 8, xy[1] + 8), f"{time:.2f}s", fill="yellow")
        sheet.save(ROOT / "media/ransac/contact_sheet.png")
    return {"path": str(VIDEO), "bytes": VIDEO.stat().st_size,
            "resolution": [width, height], "fps": fps, "duration": count / fps,
            "frames": count, "all_frames_decoded": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument("--report", action="store_true", help="Save reproducible numerical/QA report")
    args = parser.parse_args()
    result = {"numeric": validate_numeric()}
    if args.video or args.contact_sheet:
        result["video"] = validate_video(args.contact_sheet)
    output = json.dumps(result, indent=2)
    if args.report:
        (ROOT / "media/ransac").mkdir(parents=True, exist_ok=True)
        (ROOT / "media/ransac/validation.json").write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
