"""Decode every review frame; optionally export a small visual QA contact sheet."""

from pathlib import Path
import argparse
import json

import av
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "media/videos/canny_pipeline/480p15/CannyPipelinePrototype_review.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-sheet", action="store_true")
    args = parser.parse_args()
    samples = []
    targets = iter([3.3, 5.8, 8.5, 10.0, 12.3, 14.8, 16.7, 18.2, 19.2, 20.5, 22.0, 25.0])
    target = next(targets)
    count = 0
    with av.open(str(VIDEO)) as container:
        stream = container.streams.video[0]
        width, height = stream.width, stream.height
        fps = float(stream.average_rate)
        for frame in container.decode(stream):
            count += 1
            time = float(frame.pts * frame.time_base)
            if args.contact_sheet and time >= target:
                samples.append((time, frame.to_image()))
                target = next(targets, float("inf"))
    assert count > 0 and width == 854 and height == 480
    assert 24 <= count / fps <= 30, count / fps
    result = {"path": str(VIDEO), "bytes": VIDEO.stat().st_size,
              "resolution": [width, height], "fps": fps, "decoded_frames": count,
              "duration_seconds": count / fps, "readable": True}
    if args.contact_sheet:
        sheet = Image.new("RGB", (width * 3, height * 4), "#0B1020")
        draw = ImageDraw.Draw(sheet)
        for index, (time, frame) in enumerate(samples):
            xy = ((index % 3) * width, (index // 3) * height)
            sheet.paste(frame, xy)
            draw.text((xy[0] + 8, xy[1] + 8), f"{time:.2f}s", fill="yellow")
        sheet.save(ROOT / "media/refinement_contact_sheet.png")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
