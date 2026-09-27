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


if __name__ == "__main__":
    main()
