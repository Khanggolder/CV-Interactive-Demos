"""Generate compact, purpose-built teaching images (no external dataset)."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "samples"
SIZE = 512


def scene() -> Image.Image:
    image = Image.new("RGB", (SIZE, SIZE), "#b9d7ea")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 315, SIZE, SIZE), fill="#819b63")
    draw.rectangle((95, 190, 415, 375), fill="#eee4d2", outline="#34495e", width=8)
    draw.polygon([(70, 205), (255, 75), (440, 205)], fill="#b94a48", outline="#34495e")
    draw.rectangle((205, 245, 305, 375), fill="#75533b")
    draw.rectangle((125, 235, 185, 295), fill="#74b9ff", outline="#34495e", width=5)
    draw.ellipse((395, 35, 475, 115), fill="#ffd166")
    return image


def save_adjusted(name: str, scale: float, offset: float) -> None:
    base = np.asarray(scene(), dtype=np.float32)
    Image.fromarray(np.clip(base * scale + offset, 0, 255).astype(np.uint8)).save(OUTPUT / name)


def ringing_pattern() -> Image.Image:
    """Create hard discontinuities that make low-pass ringing easy to inspect."""
    image = Image.new("L", (SIZE, SIZE), 24)
    draw = ImageDraw.Draw(image)
    draw.rectangle((58, 72, 238, 250), fill=235)
    draw.rectangle((302, 118, 452, 390), fill=170)
    draw.rectangle((82, 330, 255, 430), fill=245)
    draw.line((25, 285, 485, 285), fill=105, width=10)
    return image


def interpolation_grid() -> Image.Image:
    """Create a synthetic grid with colored landmarks for geometric warps."""
    image = Image.new("RGB", (SIZE, SIZE), "#f8fafc")
    draw = ImageDraw.Draw(image)
    step = 32
    for coordinate in range(0, SIZE, step):
        width = 3 if coordinate % 128 == 0 else 1
        color = "#1e3a5f" if width == 3 else "#94a3b8"
        draw.line((coordinate, 0, coordinate, SIZE), fill=color, width=width)
        draw.line((0, coordinate, SIZE, coordinate), fill=color, width=width)
    draw.rectangle((72, 72, 184, 184), fill="#ef4444", outline="#7f1d1d", width=5)
    draw.ellipse((314, 72, 438, 196), fill="#22c55e", outline="#14532d", width=5)
    draw.polygon([(256, 280), (150, 440), (362, 440)], fill="#3b82f6", outline="#1e3a8a")
    draw.line((256, 0, 256, SIZE), fill="#e85d2a", width=5)
    draw.line((0, 256, SIZE, 256), fill="#e85d2a", width=5)
    return image


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_adjusted("low_light.png", 0.38, 0)
    save_adjusted("bright.png", 0.78, 55)
    gray = np.asarray(scene().convert("L"), dtype=np.float32)
    low_contrast = np.clip(128 + (gray - gray.mean()) * 0.28, 0, 255).astype(np.uint8)
    Image.fromarray(low_contrast).save(OUTPUT / "low_contrast.png")
    high_contrast = np.where(gray > np.median(gray), 235, 25).astype(np.uint8)
    Image.fromarray(high_contrast).save(OUTPUT / "high_contrast.png")

    yy, xx = np.indices((SIZE, SIZE))
    waves = 127.5 + 55 * np.sin(xx / 4.0) + 45 * np.sin((xx + yy) / 8.0)
    checker = ((xx // 16 + yy // 16) % 2) * 55
    texture = np.clip(waves + checker - 25, 0, 255).astype(np.uint8)
    Image.fromarray(texture).save(OUTPUT / "texture.png")

    edges = Image.new("L", (SIZE, SIZE), 24)
    draw = ImageDraw.Draw(edges)
    draw.rectangle((55, 60, 235, 240), fill=225)
    draw.ellipse((285, 70, 465, 250), fill=160)
    draw.polygon([(90, 450), (255, 275), (420, 450)], fill=245)
    draw.line((30, 270, 480, 270), fill=110, width=12)
    edges.save(OUTPUT / "clear_edges.png")

    ringing_pattern().save(OUTPUT / "ringing_pattern.png")
    interpolation_grid().save(OUTPUT / "interpolation_grid.png")


if __name__ == "__main__":
    main()
