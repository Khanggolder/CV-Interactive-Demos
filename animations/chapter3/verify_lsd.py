"""Numerical audit and complete decode of the self-contained LSD prototype."""

import argparse
import json
from decimal import Decimal, localcontext
from math import comb

import numpy as np

from lsd import (
    DATA_SEED, EPSILON, MIN_MAGNITUDE, ROOT, SHAPE, TAU, angular_difference_pi,
    experiment, neighbors, position_xy, select_seed,
)

VIDEO = ROOT / "media/videos/lsd/480p15/LSDLineSegmentDetector_review.mp4"


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def validate():
    data = experiment()
    assert np.isclose(angular_difference_pi(np.deg2rad(5), np.deg2rad(175)), np.deg2rad(10))
    gx, gy = data["gx"], data["gy"]
    tangent = np.stack((np.cos(data["angles"]), np.sin(data["angles"])), axis=-1)
    assert np.max(np.abs(tangent[..., 0] * gx + tangent[..., 1] * gy)) < 1e-12
    used = np.zeros(SHAPE, dtype=bool)
    for region in data["regions"]:
        assert region["seed"] == select_seed(data["magnitude"], used)
        members = set(region["pixels"])
        reached, queue = {region["seed"]}, [region["seed"]]
        while queue:
            for pixel in neighbors(queue.pop()):
                if pixel in members and pixel not in reached:
                    reached.add(pixel)
                    queue.append(pixel)
        assert reached == members
        for pixel in members:
            assert not used[pixel]
            assert data["magnitude"][pixel] >= MIN_MAGNITUDE
            assert angular_difference_pi(data["angles"][pixel], region["reference"]) < TAU
            used[pixel] = True
        for event in region["events"]:
            assert event["parent"] in members
            assert event["pixel"] in list(neighbors(event["parent"]))
            assert event["accepted"] == (event["difference"] < TAU)
            assert event["accepted"] == (event["pixel"] in members)
        fit = region["fit"]
        pixels = np.array(region["pixels"])
        theta = data["angles"][tuple(pixels.T)]
        # Dot-product comparison independently checks the directed alignment.
        direct = np.cos(theta - fit["angle"]) > np.cos(TAU)
        assert np.array_equal(direct, fit["aligned"])
        assert fit["k"] == int(direct.sum())
        if fit["n"] > 1:
            delta = fit["endpoints"][1] - fit["endpoints"][0]
            assert angular_difference_pi(np.arctan2(delta[1], delta[0]), fit["angle"]) < 1e-12
            xy = pixels[:, ::-1]
            weights = data["magnitude"][tuple(pixels.T)]
            covariance = ((xy - fit["center"]) * weights[:, None]).T @ (xy - fit["center"])
            direction = np.array([np.cos(fit["angle"]), np.sin(fit["angle"])])
            assert np.isclose(direction @ covariance @ direction, np.linalg.eigvalsh(covariance)[-1])
        with localcontext() as context:
            context.prec = 65
            p = Decimal(1) / 8
            tail = sum(Decimal(comb(fit["n"], k)) * p ** k * (1 - p) ** (fit["n"] - k)
                       for k in range(fit["k"], fit["n"] + 1))
            assert np.isclose(region["nfa"], float(tail * data["tests"]), rtol=1e-12, atol=0)
        assert region["accept"] == (region["nfa"] < EPSILON)
        for key in ("center", "endpoints", "rectangle"):
            assert np.isfinite(fit[key]).all()
    assert select_seed(data["magnitude"], used) is None
    for key in ("intensity", "gx", "gy", "magnitude", "angles"):
        assert np.isfinite(data[key]).all()
    strong, noisy = data["regions"][:2]
    assert strong["accept"] and not noisy["accept"]
    assert (strong["fit"]["n"], strong["fit"]["k"], noisy["fit"]["n"], noisy["fit"]["k"]) == (33, 33, 4, 2)
    for region in (strong, noisy):
        for xy in region["fit"]["rectangle"]:
            screen = position_xy(xy)
            assert abs(screen[0]) < 7.9 and abs(screen[1]) < 4.4
    return {"dimensions_y_x": SHAPE, "coordinates": "[y,x], zero based, y increases upward",
            "data_seed": DATA_SEED, "tau_degrees": float(np.rad2deg(TAU)), "p0": float(TAU / np.pi),
            "epsilon": EPSILON, "N_T": data["tests"], "shown_region_indices": [0, 1],
            "intensity": data["intensity"], "gradient_x": gx, "gradient_y": gy,
            "gradient_magnitude": data["magnitude"], "level_line_angle_radians": data["angles"],
            "regions": data["regions"], "checks_passed": True,
            "limitations": ["Teaching prototype, not the reference LSD implementation.",
                            "Chapter-3 NFA uses LSR pixels and the number of grown LSRs, not all rectangle tests.",
                            "Regions are adaptively selected: displayed NFA is not a calibrated whole-image false-alarm guarantee.",
                            "Growth is modulo pi; NFA retains directed gradient polarity so p0=tau/pi is consistent.",
                            "Two candidates shown; other computed regions are not animated."]}


def video_check(qa=False):
    import av
    from PIL import Image, ImageDraw
    samples = []
    targets = iter([2, 5, 8, 11, 15, 18, 21, 24, 27, 30, 32, 34])
    target = next(targets)
    with av.open(str(VIDEO)) as container:
        stream = container.streams.video[0]
        width, height, fps = stream.width, stream.height, float(stream.average_rate)
        count = 0
        for frame in container.decode(stream):
            count += 1
            seconds = float(frame.pts * frame.time_base)
            if qa and seconds >= target:
                samples.append((seconds, frame.to_image()))
                target = next(targets, float("inf"))
    assert (width, height, fps) == (854, 480, 15)
    assert 30 <= count / fps <= 38, count / fps
    if qa:
        sheet = Image.new("RGB", (width * 3, height * 4), "#0B1020")
        draw = ImageDraw.Draw(sheet)
        for i, (time, frame) in enumerate(samples):
            xy = (i % 3 * width, i // 3 * height)
            sheet.paste(frame, xy)
            draw.text((xy[0]+8, xy[1]+8), f"{time:.2f}s", fill="yellow")
        sheet.save(ROOT / "media/lsd/contact_sheet.png")
    return {"path": str(VIDEO), "bytes": VIDEO.stat().st_size, "width": width, "height": height,
            "fps": fps, "frames": count, "duration": count / fps, "all_frames_decoded": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--qa", action="store_true")
    args = parser.parse_args()
    report = {"numerical": validate()}
    if args.video or args.qa:
        report["video"] = video_check(args.qa)
    output = ROOT / "media/lsd"
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation.json").write_text(json.dumps(report, default=serial, indent=2) + "\n", encoding="utf-8")
    for i, region in enumerate(report["numerical"]["regions"][:2]):
        print(f"Candidate {i}: seed={region['seed']}, n={region['fit']['n']}, k={region['fit']['k']}, NFA={region['nfa']}, accept={region['accept']}")
    print(json.dumps(report.get("video", {"numerical_checks": "PASS"}), indent=2))
