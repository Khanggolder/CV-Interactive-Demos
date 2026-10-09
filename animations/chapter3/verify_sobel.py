"""Independent arithmetic, scene-layout preflight, and complete video decode."""

import argparse
import json
import math

import numpy as np

from sobel_convolution import (
    KERNEL_X, KERNEL_Y, PATCH_CENTER, ROOT, VECTOR_SCALE, SobelConvolution,
    compute_sobel, generate_sobel_example, screen_vector,
)

VIDEO = ROOT / "media/videos/sobel_convolution/480p15/SobelConvolution_review.mp4"
OUTPUT = ROOT / "media/sobel"


def validate():
    image = generate_sobel_example()
    data = compute_sobel(image)
    patch = data["patch"]
    np.testing.assert_array_equal(patch, [[20, 20, 200]] * 3)
    np.testing.assert_array_equal(patch, image[1:4, 2:5])
    np.testing.assert_array_equal(KERNEL_X, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    np.testing.assert_array_equal(KERNEL_Y, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    multiplications = []
    for name, kernel in (("x", KERNEL_X), ("y", KERNEL_Y)):
        total = 0
        for r in range(3):
            for c in range(3):
                value, weight = int(patch[r, c]), int(kernel[r, c])
                product = value * weight
                assert product == data[f"products_{name}"][r, c]
                total += product
                multiplications.append(dict(axis=name, row=r, column=c,
                                            value=value, weight=weight, product=product))
        assert total == data[f"g{name}"]
    # Independent separable weighted differences.
    gx = sum(w * int(patch[r, 2] - patch[r, 0]) for r, w in enumerate([1, 2, 1]))
    gy = sum(w * int(patch[2, c] - patch[0, c]) for c, w in enumerate([1, 2, 1]))
    assert (gx, gy) == (data["gx"], data["gy"]) == (720, 0)
    assert math.sqrt(gx * gx + gy * gy) == data["magnitude"] == 720
    assert math.atan2(gy, gx) == data["theta_radians"] == 0
    assert math.degrees(math.atan2(gy, gx)) == data["theta_degrees"] == 0
    vector = screen_vector(gx, gy)
    np.testing.assert_allclose(vector, [3.1, 0, 0])
    assert vector @ np.array([0, 1, 0]) == 0  # Vertical boundary tangent.
    assert np.isfinite(np.concatenate([image.ravel(), vector, list(data[k] for k in
        ("gx", "gy", "magnitude", "theta_radians", "theta_degrees"))])).all()
    shifted = compute_sobel(image, (2, 4))
    np.testing.assert_array_equal(shifted["patch"], [[20, 200, 200]] * 3)
    assert (shifted["gx"], shifted["gy"]) == (720, 0)
    return dict(source_image=image.tolist(), patch_center=list(PATCH_CENTER),
                kernel_x=KERNEL_X.tolist(), kernel_y=KERNEL_Y.tolist(),
                **{key: value.tolist() if isinstance(value, np.ndarray) else value
                   for key, value in data.items()}, multiplications=multiplications,
                shifted_patch=shifted["patch"].tolist(), shifted_gx=shifted["gx"], shifted_gy=shifted["gy"],
                kernel_convention="Direct displayed-mask correlation; kernels are NOT flipped.",
                coordinates="Image x right, y down; display vector = scale * (Gx, -Gy).",
                visualization=dict(vector_scale=VECTOR_SCALE, image_vector_scale=VECTOR_SCALE * 0.62,
                                   intensity_color="Linear value/255 blue-gray mapping; arithmetic unnormalized.",
                                   triangle="Gy=0: triangle collapsed, theta rays coincide (0 degrees)."),
                arithmetic_checks_passed=True, edge_gradient_perpendicular=True)


def check_video(qa=False):
    import av
    from PIL import Image, ImageDraw
    samples = []
    targets = iter([2.8, 4.8, 7.7, 9, 11, 12.5, 15.8, 18.5, 20.5, 22.8, 25, 27])
    target = next(targets)
    with av.open(str(VIDEO)) as container:
        stream = container.streams.video[0]
        width, height, fps = stream.width, stream.height, float(stream.average_rate)
        count = 0
        last_time = -1
        for frame in container.decode(stream):
            time = float(frame.pts * frame.time_base)
            assert time > last_time
            last_time = time
            count += 1
            if qa and time >= target:
                samples.append((time, frame.to_image()))
                target = next(targets, float("inf"))
        assert stream.frames == count
    assert (width, height, fps) == (854, 480, 15)
    assert 24 <= count / fps <= 32, count / fps
    if qa:
        sheet = Image.new("RGB", (width * 3, height * 4), "#0B1020")
        draw = ImageDraw.Draw(sheet)
        for i, (seconds, frame) in enumerate(samples):
            xy = (i % 3 * width, i // 3 * height)
            sheet.paste(frame, xy)
            draw.text((xy[0] + 8, xy[1] + 8), f"{seconds:.2f}s", fill="yellow")
            if i in (1, 8):
                frame.save(OUTPUT / f"review_{seconds:.2f}s.png")
        sheet.save(OUTPUT / "contact_sheet.png")
    return dict(path=str(VIDEO), bytes=VIDEO.stat().st_size, duration=count/fps,
                width=width, height=height, fps=fps, frames=count, all_frames_decoded=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--qa", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"numerical": validate()}
    existing = OUTPUT / "validation.json"
    if existing.exists():
        report.update({k: v for k, v in json.loads(existing.read_text(encoding="utf-8")).items()
                       if k != "numerical"})
    if args.preflight:
        from manim import tempconfig
        with tempconfig(dict(dry_run=True, skip_animations=True, write_to_movie=False,
                             save_last_frame=False, pixel_width=854, pixel_height=480,
                             frame_rate=15, progress_bar="none")):
            scene = SobelConvolution()
            scene.render()
        report["layout_preflight"] = "PASS: finite geometry and frame bounds after every play."
    if args.video or args.qa:
        report["video"] = check_video(args.qa)
    existing.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arithmetic": "PASS", "Gx": 720, "Gy": 0, "M": 720, "theta": 0,
                      "layout": report.get("layout_preflight"), "video": report.get("video")}, indent=2))
