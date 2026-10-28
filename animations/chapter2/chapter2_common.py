"""Shared visual helpers for the Chapter 2 Manim teaching scenes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    FadeIn, FadeOut, Group as VGroup, ImageMobject, ManimColor, Rectangle,
    Succession, Text, config, interpolate_color,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WEBAPP = REPO / "chapter2_webapp"
if str(WEBAPP) not in sys.path:
    sys.path.insert(0, str(WEBAPP))

BG = "#0B1020"
WHITE = "#F8FAFC"
MUTED = "#8892A8"
GRID = "#283248"
CYAN = "#47C2FF"
ORANGE = "#FF8C42"
YELLOW = "#FFD166"
GREEN = "#69D394"
PURPLE = "#B392F0"
RED = "#E06C75"

config.background_color = BG
config.frame_width = 16
config.frame_height = 9
config.media_dir = str(HERE / "media")
config.text_dir = str(HERE / "media/chapter2_texts")


def text(value: str, xy=(0.0, 0.0), color=WHITE, size=28, weight="NORMAL") -> Text:
    return Text(value, font_size=size, color=color, weight=weight).move_to([xy[0], xy[1], 0])


class TextSwap(Succession):
    """Remove a complete old caption before adding the next one."""

    def __init__(self, old, new, out_time=0.18, in_time=0.24):
        super().__init__(FadeOut(old, run_time=out_time), FadeIn(new, run_time=in_time))

    def next_animation(self):
        if self.active_index == 0:
            self.active_animation.finish()
            self.scene.remove(self.active_animation.mobject)
            self.update_active_animation(1)
        else:
            super().next_animation()

    def interpolate(self, alpha):
        progress = alpha * max(self.run_time, self.max_end_time) / self.max_end_time
        super().interpolate(min(progress, 1))


def gray_rgba(values: np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    lo, hi = float(np.min(a)), float(np.max(a))
    normalized = np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)
    gray = np.clip(25 + 220 * normalized, 0, 255).astype(np.uint8)
    return np.dstack([gray, gray, gray, np.full_like(gray, 255)])


def signed_rgba(values: np.ndarray, robust=True) -> np.ndarray:
    """Blue negative, dark zero, orange positive; sign is never discarded."""
    a = np.asarray(values, dtype=float)
    scale = float(np.percentile(np.abs(a), 99)) if robust else float(np.max(np.abs(a)))
    scale = max(scale, 1e-12)
    z = np.clip(a / scale, -1, 1)
    base = np.array([11, 16, 32], dtype=float)
    pos = np.array([255, 140, 66], dtype=float)
    neg = np.array([71, 194, 255], dtype=float)
    rgb = np.empty((*a.shape, 3), dtype=np.uint8)
    for channel in range(3):
        rgb[..., channel] = np.where(
            z >= 0,
            base[channel] + z * (pos[channel] - base[channel]),
            base[channel] + (-z) * (neg[channel] - base[channel]),
        ).astype(np.uint8)
    return np.dstack([rgb, np.full(a.shape, 255, dtype=np.uint8)])


def positive_rgba(values: np.ndarray, accent=CYAN) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    lo, hi = float(np.min(a)), float(np.max(a))
    z = np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)
    start, end = ManimColor(BG), ManimColor(accent)
    colors = np.empty((*a.shape, 4), dtype=np.uint8)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            rgb = interpolate_color(start, end, float(z[r, c])).to_rgb()
            colors[r, c, :3] = np.round(255 * np.array(rgb)).astype(np.uint8)
            colors[r, c, 3] = 255
    return colors


def image_mobject(values: np.ndarray, *, width=4.0, signed=False, accent=CYAN) -> ImageMobject:
    rgba = signed_rgba(values) if signed else gray_rgba(values)
    mob = ImageMobject(rgba)
    mob.set_resampling_algorithm(0)
    mob.set(width=width)
    return mob


def framed_image(values: np.ndarray, title: str, center, width=4.0, signed=False,
                 accent=CYAN, title_size=25) -> VGroup:
    image = image_mobject(values, width=width, signed=signed, accent=accent).move_to(center)
    frame = Rectangle(width=image.width + 0.08, height=image.height + 0.08,
                      stroke_color=GRID, stroke_width=1.5).move_to(image)
    label = text(title, (image.get_x(), image.get_top()[1] + 0.34), accent, title_size)
    return VGroup(image, frame, label)


def pixel_grid(values: np.ndarray, center, cell=0.42, decimals=2, signed=False,
               font_size=None, accent=CYAN) -> VGroup:
    a = np.asarray(values, dtype=float)
    h, w = a.shape
    scale = max(float(np.max(np.abs(a))), 1e-12)
    group = VGroup()
    for r in range(h):
        for c in range(w):
            value = float(a[r, c])
            if signed:
                color = ORANGE if value > 0 else CYAN if value < 0 else BG
                opacity = 0.12 + 0.72 * abs(value) / scale
            else:
                color = interpolate_color(ManimColor(BG), ManimColor(accent),
                                          float(np.clip(value / scale, 0, 1)))
                opacity = 0.9
            pos = np.array(center) + cell * np.array([c-(w-1)/2, (h-1)/2-r, 0])
            box = Rectangle(width=cell, height=cell, stroke_color=GRID, stroke_width=1,
                            fill_color=color, fill_opacity=opacity).move_to(pos)
            if decimals == 0:
                label_value = f"{value:.0f}"
            else:
                label_value = f"{value:.{decimals}f}"
            number = text(label_value, (pos[0], pos[1]), WHITE,
                          font_size or max(13, int(20*cell/.42)))
            group.add(VGroup(box, number))
    return group


def check_scene_bounds(scene) -> None:
    for root in scene.mobjects:
        for mob in root.get_family():
            if mob.has_points():
                points = mob.get_all_points()
                assert np.isfinite(points).all()
                assert mob.get_left()[0] > -8.02 and mob.get_right()[0] < 8.02
                assert mob.get_bottom()[1] > -4.52 and mob.get_top()[1] < 4.52


def progress_bar(names: list[str], y=-4.02):
    xs = np.linspace(-6.6, 6.6, len(names))
    labels = VGroup(*[text(name, (float(x), y), MUTED, 18) for x, name in zip(xs, names)])
    return labels, xs
