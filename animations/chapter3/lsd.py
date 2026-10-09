"""LSD teaching prototype: measured gradients, connected LSRs, computed NFA.

This follows Chapter 3's LSR-based binomial example, not the full reference
LSD rectangle refinement/testing procedure. Growth uses axial (modulo pi)
orientation; NFA alignment retains gradient polarity (modulo 2*pi), for which
the prescribed p0 = tau/pi is the directed-uniform null probability.
"""

from __future__ import annotations

from collections import deque
from math import comb, fsum
from pathlib import Path

import numpy as np
from manim import (
    Arc, Arrow, Circle, Create, FadeIn, FadeOut, Indicate, LaggedStart, Line,
    ManimColor, Polygon, Rectangle, Rotate, Scene, Succession, Text, Transform, VGroup,
    config, interpolate_color,
)


ROOT = Path(__file__).resolve().parent
DATA_SEED = 9
SHAPE = (10, 12)  # Arrays indexed [y, x], with Cartesian y increasing upward.
TAU = np.deg2rad(22.5)
EPSILON = 1.0
MIN_MAGNITUDE = 0.09
BACKGROUND, MUTED, GRID = "#0B1020", "#8892A8", "#283248"
WHITE, GRADIENT, LEVEL = "#F8FAFC", "#B392F0", "#47C2FF"
ACTIVE, RESULT, REJECT = "#FFD166", "#69D394", "#FF8C42"
CELL = 0.60
GRID_CENTER = np.array([-1.9, 0.1, 0.0])
config.background_color = BACKGROUND
config.frame_width, config.frame_height = 16, 9
config.media_dir = str(ROOT / "media")
config.text_dir = str(ROOT / "media/lsd/texts")


def generate_synthetic_field():
    y, x = np.indices(SHAPE, dtype=float)
    rng = np.random.default_rng(DATA_SEED)
    edge = 0.5 + 0.42 * np.tanh((y - 0.55 * x - 1.3) / 0.8)
    blob = 0.40 * np.exp(-((x - 8.8) ** 2 + (y - 1.6) ** 2) / (2 * 0.75 ** 2))
    return edge + blob + rng.normal(0, 0.055, SHAPE)


def compute_gradient_field(intensity):
    gy, gx = np.gradient(intensity)
    return gx, gy, np.hypot(gx, gy)


def gradient_angle(gx, gy):
    return np.arctan2(gy, gx)


def level_line_angle(gx, gy):
    # Clockwise 90 degrees: preserves polarity for NFA validation.
    return (gradient_angle(gx, gy) - np.pi / 2) % (2 * np.pi)


def angular_difference_pi(a, b):
    return np.abs((np.asarray(a) - b + np.pi / 2) % np.pi - np.pi / 2)


def angular_difference_2pi(a, b):
    return np.abs((np.asarray(a) - b + np.pi) % (2 * np.pi) - np.pi)


def select_seed(magnitude, used):
    eligible = (~used) & (magnitude >= MIN_MAGNITUDE)
    if not eligible.any():
        return None
    return tuple(map(int, np.unravel_index(np.argmax(np.where(eligible, magnitude, -np.inf)), SHAPE)))


def neighbors(pixel):
    y, x = pixel
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy or dx) and 0 <= y + dy < SHAPE[0] and 0 <= x + dx < SHAPE[1]:
                yield y + dy, x + dx


def grow_lsr(seed, angles, magnitude, used):
    """8-connected BFS; fixed seed reference makes every decision auditable."""
    reference = float(angles[seed])
    queue, visited, members = deque([seed]), {seed}, [seed]
    used[seed] = True
    events = []
    while queue:
        parent = queue.popleft()
        for point in neighbors(parent):
            if point in visited or used[point] or magnitude[point] < MIN_MAGNITUDE:
                continue
            visited.add(point)
            difference = float(angular_difference_pi(angles[point], reference))
            accepted = difference < TAU
            events.append({"parent": parent, "pixel": point,
                           "difference": difference, "accepted": bool(accepted)})
            if accepted:
                used[point] = True
                queue.append(point)
                members.append(point)
    return {"seed": seed, "reference": reference, "pixels": members, "events": events}


def fit_lsr_segment(region, magnitude, angles):
    pixels = np.array(region["pixels"])
    xy = pixels[:, ::-1].astype(float)
    weights = magnitude[tuple(pixels.T)]
    center = np.average(xy, axis=0, weights=weights)
    if len(xy) > 1:
        covariance = ((xy - center) * weights[:, None]).T @ (xy - center)
        _, vectors = np.linalg.eigh(covariance)
        tangent = vectors[:, -1]
    else:
        tangent = np.array([np.cos(region["reference"]), np.sin(region["reference"])])
    # Choose the directed PCA axis closest to the observed level-line polarity.
    direction_sum = np.sum(weights[:, None] * np.column_stack(
        (np.cos(angles[tuple(pixels.T)]), np.sin(angles[tuple(pixels.T)]))), axis=0)
    if tangent @ direction_sum < 0:
        tangent *= -1
    normal = np.array([-tangent[1], tangent[0]])
    along, across = (xy - center) @ tangent, (xy - center) @ normal
    # Enclose every pixel's square, not just its center, in the fitted box.
    pad_t = 0.5 * np.abs(tangent).sum()
    pad_n = 0.5 * np.abs(normal).sum()
    amin, amax = along.min() - pad_t, along.max() + pad_t
    bmin, bmax = across.min() - pad_n, across.max() + pad_n
    rectangle = np.array([center + a * tangent + b * normal
                          for a, b in ((amin, bmin), (amax, bmin), (amax, bmax), (amin, bmax))])
    endpoints = np.array([center + along.min() * tangent, center + along.max() * tangent])
    angle = float(np.arctan2(tangent[1], tangent[0]) % (2 * np.pi))
    aligned = angular_difference_2pi(angles[tuple(pixels.T)], angle) < TAU
    return {"center": center, "endpoints": endpoints, "rectangle": rectangle,
            "angle": angle, "aligned": aligned, "n": len(xy), "k": int(aligned.sum())}


def compute_binomial_tail(n, k, p):
    if not 0 <= k <= n or not 0 <= p <= 1:
        raise ValueError("Invalid binomial parameters")
    return fsum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


def compute_nfa(n, k, tests, p0=TAU / np.pi):
    return tests * compute_binomial_tail(n, k, p0)


def experiment():
    intensity = generate_synthetic_field()
    gx, gy, magnitude = compute_gradient_field(intensity)
    angles = level_line_angle(gx, gy)
    used = np.zeros(SHAPE, dtype=bool)
    regions = []
    while (seed := select_seed(magnitude, used)) is not None:
        region = grow_lsr(seed, angles, magnitude, used)
        region["fit"] = fit_lsr_segment(region, magnitude, angles)
        regions.append(region)
    # Every grown candidate in the full image is counted, not only those shown.
    for region in regions:
        fit = region["fit"]
        region["nfa"] = compute_nfa(fit["n"], fit["k"], len(regions))
        region["accept"] = region["nfa"] < EPSILON
    return {"intensity": intensity, "gx": gx, "gy": gy, "magnitude": magnitude,
            "angles": angles, "regions": regions, "tests": len(regions)}


def position_xy(xy):
    return GRID_CENTER + CELL * np.array([xy[0] - 5.5, xy[1] - 4.5, 0])


def position(pixel):
    return position_xy(pixel[::-1])


def label(text, xy, color=WHITE, size=27, bold=False):
    return Text(text, font_size=size, color=color,
                weight="BOLD" if bold else "NORMAL").move_to([*xy, 0])


class TextSwap(Succession):
    """Whole-sentence fades; explicitly remove old text before introducing new."""

    def __init__(self, old, new):
        super().__init__(FadeOut(old, run_time=0.20), FadeIn(new, run_time=0.25))

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


def create_grid_visual(intensity):
    normalized = (intensity - intensity.min()) / np.ptp(intensity)
    return VGroup(*[Rectangle(width=CELL, height=CELL, stroke_color=GRID,
                             stroke_width=0.7, fill_opacity=1,
                             fill_color=interpolate_color(ManimColor("#10192B"), ManimColor("#53677F"),
                                                          float(normalized[y, x])))
                   .move_to(position((y, x)))
                   for y in range(SHAPE[0]) for x in range(SHAPE[1])])


def create_orientation_arrow(pixel, gx, gy):
    vector = np.array([gx[pixel], gy[pixel], 0.0])
    vector /= np.linalg.norm(vector)
    center = position(pixel)
    return Arrow(center - 0.18 * vector, center + 0.18 * vector, buff=0,
                 color=GRADIENT, stroke_width=2.5, tip_length=0.08).set_z_index(5)


def create_level_line(pixel, angle):
    vector = np.array([np.cos(angle), np.sin(angle), 0])
    center = position(pixel)
    return Line(center - 0.20 * vector, center + 0.20 * vector,
                color=LEVEL, stroke_width=2.6).set_z_index(5)


def candidate_visual(region):
    fit = region["fit"]
    box = Polygon(*map(position_xy, fit["rectangle"]), color=ACTIVE,
                  stroke_width=1.5, fill_opacity=0).set_z_index(6)
    line = Line(*map(position_xy, fit["endpoints"]), color=ACTIVE,
                stroke_width=3.6).set_z_index(7)
    return box, line


class LSDLineSegmentDetector(Scene):
    """Measured directions become connected support, then validated segments."""

    def play(self, *animations, **kwargs):
        super().play(*animations, **kwargs)
        for root in self.mobjects:
            for mob in root.get_family():
                if mob.has_points():
                    assert np.isfinite(mob.get_all_points()).all()
                    assert mob.get_left()[0] > -7.95 and mob.get_right()[0] < 7.95
                    assert mob.get_bottom()[1] > -4.45 and mob.get_top()[1] < 4.45

    def swap(self, name, target):
        old = getattr(self, name)
        setattr(self, name, target)
        return TextSwap(old, target)

    def caption_change(self, text, color=WHITE):
        target = label(text, (0, -3.85), color, 28)
        assert target.get_top()[1] < -3.5
        return self.swap("caption", target)

    def stage_change(self, text, color=WHITE):
        return self.swap("stage", label(text, (4.8, 2.55), color, 30, True))

    def cell(self, pixel):
        return self.grid[pixel[0] * SHAPE[1] + pixel[1]]

    def construct(self):
        self.data = experiment()
        self.strong, self.noisy = self.data["regions"][:2]
        assert self.strong["accept"] and not self.noisy["accept"]
        self.grid = create_grid_visual(self.data["intensity"])
        self.levels = {}
        self.caption = label("Nearby gradients align", (0, -3.85), size=28)
        self.stage = label("Gradient", (4.8, 2.55), GRADIENT, 30, True)
        self.add(label("LSD · Line Segment Detector", (0, 3.95), size=40, bold=True))
        self.introduce_gradients()
        self.construct_level_lines()
        self.select_strong_seed()
        self.grow_strong_region()
        self.fit_strong_region()
        self.show_noisy_region()
        self.validate_candidates()
        self.final_result()

    def introduce_gradients(self):
        self.play(FadeIn(self.grid), FadeIn(self.stage), FadeIn(self.caption), run_time=0.7)
        # Only nine actual high-magnitude pixels first, not a wall of arrows.
        self.representatives = self.strong["pixels"][:9]
        self.arrows = {p: create_orientation_arrow(p, self.data["gx"], self.data["gy"])
                       for p in self.representatives}
        self.play(LaggedStart(*[Create(a) for a in self.arrows.values()], lag_ratio=0.12), run_time=1.1)
        self.wait(0.5)

    def construct_level_lines(self):
        seed = self.strong["seed"]
        angle = self.data["angles"][seed]
        normal = np.array([-np.sin(angle), np.cos(angle), 0])
        tangent = np.array([np.cos(angle), np.sin(angle), 0])
        center = np.array([4.7, 0.8, 0])
        self.seed_ring = Circle(radius=0.27, color=ACTIVE, stroke_width=2).move_to(position(seed)).set_z_index(8)
        arrow = Arrow(center, center + normal * 1.0, color=GRADIENT,
                      buff=0, stroke_width=3, tip_length=0.13)
        rotating = Line(center - 0.85 * normal, center + 0.85 * normal, color=LEVEL, stroke_width=3)
        gradient_label = label("∇I(p)", (6.05, 1.65), GRADIENT, 26)
        self.demo = VGroup(arrow, rotating, gradient_label)
        self.play(Create(self.seed_ring), Create(arrow), FadeIn(gradient_label),
                  Create(rotating), run_time=0.55)
        self.play(Rotate(rotating, angle=-np.pi / 2, about_point=center),
                  self.stage_change("Level line", LEVEL), run_time=0.85)
        corner = VGroup(Line(center + normal * 0.20, center + (normal + tangent) * 0.20, color=MUTED),
                        Line(center + (normal + tangent) * 0.20, center + tangent * 0.20, color=MUTED))
        perpendicular = label("90° to the gradient", (4.8, -0.6), size=25)
        self.demo.add(corner, perpendicular)
        self.play(Create(corner), FadeIn(perpendicular),
                  self.caption_change("Turn 90° → follow the edge", LEVEL), run_time=0.5)
        for pixel, arrow in self.arrows.items():
            self.levels[pixel] = arrow
        self.play(*[Transform(self.arrows[p], create_level_line(p, self.data["angles"][p]))
                    for p in self.representatives], run_time=0.85)
        self.wait(1.5)

    def select_strong_seed(self):
        self.play(FadeOut(self.demo), self.stage_change("Seed", ACTIVE),
                  self.caption_change("Start at the strongest unused gradient", ACTIVE),
                  self.cell(self.strong["seed"]).animate.set_fill(ACTIVE, opacity=0.45), run_time=0.5)
        self.info = label("Largest |∇I|", (4.8, 1.65), ACTIVE, 26)
        self.play(FadeIn(self.info), Indicate(self.seed_ring, color=ACTIVE, scale_factor=1.2), run_time=0.55)
        self.wait(0.45)

    def animate_region_growth(self, region, duration):
        # Each accepted BFS event adds exactly that neighboring pixel; a
        # rejected event flashes only its already-computed incompatible cell.
        self.cell(region["seed"]).set_fill(RESULT, opacity=0.38)
        events = region["events"]
        accepted = [e for e in events if e["accepted"]]
        rejected = [e for e in events if not e["accepted"]]
        # Every acceptance is shown; up to four rejected comparisons make the
        # contrast readable without overwhelming the growth with flashes.
        shown_rejects = {e["pixel"] for e in rejected[:4]}
        shown = [e for e in events if e["accepted"] or e["pixel"] in shown_rejects]
        per_event = duration / max(1, len(shown))
        for event in shown:
            pixel = event["pixel"]
            if pixel not in self.levels:
                self.levels[pixel] = create_level_line(pixel, self.data["angles"][pixel])
                self.add(self.levels[pixel])
            if event["accepted"]:
                self.play(self.cell(pixel).animate.set_fill(RESULT, opacity=0.38),
                          self.levels[pixel].animate.set_color(LEVEL), run_time=per_event)
            else:
                self.play(Indicate(self.levels[pixel], color=REJECT, scale_factor=1.4),
                          run_time=per_event)
        assert len(accepted) + 1 == len(region["pixels"])

    def grow_strong_region(self):
        self.play(self.stage_change("Grow", RESULT),
                  self.swap("info", label("τθ = 22.5°", (4.8, 1.65), ACTIVE, 28)),
                  self.caption_change("Similar neighboring directions join the region"), run_time=0.5)
        self.rule = label("|Δθ| < τθ", (4.8, 0.95), size=28)
        self.explanation = label("Cyan: join\nOrange flash: reject", (4.8, -0.1), MUTED, 25)
        self.play(FadeIn(self.rule), FadeIn(self.explanation), run_time=0.35)
        self.animate_region_growth(self.strong, duration=4.0)
        self.wait(1.3)

    def fit_strong_region(self):
        self.box_a, self.segment_a = candidate_visual(self.strong)
        self.play(self.stage_change("LSR", LEVEL),
                  self.swap("info", label("Line Support Region", (4.8, 1.65), LEVEL, 25)),
                  FadeOut(self.rule), FadeOut(self.explanation), FadeOut(self.seed_ring),
                  self.caption_change("An elongated region supports a segment"), run_time=0.5)
        self.play(Create(self.box_a), run_time=0.65)
        self.play(Create(self.segment_a), self.stage_change("Candidate A", ACTIVE), run_time=0.65)
        self.wait(1.25)

    def show_noisy_region(self):
        seed = self.noisy["seed"]
        self.seed_ring.move_to(position(seed))
        self.play(FadeIn(self.seed_ring), self.stage_change("Another seed", ACTIVE),
                  self.swap("info", label("Same growth rule", (4.8, 1.65), MUTED, 25)),
                  self.caption_change("Short, noisy support can also form a candidate"), run_time=0.5)
        if seed not in self.levels:
            self.levels[seed] = create_level_line(seed, self.data["angles"][seed])
            self.add(self.levels[seed])
        self.animate_region_growth(self.noisy, duration=0.85)
        self.box_b, self.segment_b = candidate_visual(self.noisy)
        self.play(Create(self.box_b), Create(self.segment_b), FadeOut(self.seed_ring),
                  self.stage_change("Candidate B", ACTIVE), run_time=0.6)
        self.wait(0.45)

    def validate_candidates(self):
        self.play(self.stage_change("NFA", WHITE),
                  self.swap("info", label("Number of False Alarms", (4.8, 1.85), MUTED, 23)),
                  self.caption_change("Could this alignment happen by chance?"), run_time=0.5)
        self.cards = []
        for name, region, y in (("A", self.strong, 0.85), ("B", self.noisy, -0.9)):
            fit = region["fit"]
            heading = label(f"{name}:  n = {fit['n']}   k = {fit['k']}", (4.8, y), size=27)
            # One short tick per measured pixel; cyan means aligned with the
            # fitted directed segment, not just accepted during seed growth.
            ticks = VGroup(*[Line([3.4 + (i % 17) * 0.17, y - 0.40 - (i // 17) * 0.19, 0],
                                  [3.4 + (i % 17) * 0.17, y - 0.28 - (i // 17) * 0.19, 0],
                                  color=LEVEL if aligned else REJECT, stroke_width=2.5)
                            for i, aligned in enumerate(fit["aligned"])])
            self.cards.append(VGroup(heading, ticks))
        self.play(FadeIn(self.cards[0]), FadeIn(self.cards[1]), run_time=0.55)
        # The noisy seed region can pass the growth test yet disagree with
        # its fitted geometric axis; show exactly which support pixels do so.
        self.play(*[self.levels[p].animate.set_color(LEVEL if keep else REJECT)
                    for p, keep in zip(self.noisy["pixels"], self.noisy["fit"]["aligned"])],
                  self.caption_change("More aligned pixels → less likely by chance", LEVEL), run_time=0.6)
        self.wait(1.5)
        self.test_note = label(f"{self.data['tests']} tested LSRs · 2 shown", (4.8, -2.7), MUTED, 22)
        self.play(FadeIn(self.test_note),
                  self.swap("info", label("NFA = tests × chance", (4.8, 1.85), size=25)), run_time=0.5)
        self.nfa_labels = []
        self.decisions = []
        for region, y in ((self.strong, -0.03), (self.noisy, -1.65)):
            value = f"{region['nfa']:.2e}" if region["nfa"] < 0.01 else f"{region['nfa']:.3f}"
            value_label = label(f"NFA = {value}", (4.8, y), ACTIVE, 25)
            self.nfa_labels.append(value_label)
        self.play(*[FadeIn(mob) for mob in self.nfa_labels],
                  self.caption_change("Accept when NFA < 1", ACTIVE), run_time=0.5)
        self.wait(0.7)
        self.play(self.caption_change("Fewer than one expected false alarm per image"), run_time=0.5)
        for region, segment, box, card, value_label, y in zip(
                (self.strong, self.noisy), (self.segment_a, self.segment_b),
                (self.box_a, self.box_b), self.cards, self.nfa_labels, (0.85, -0.9)):
            color = RESULT if region["accept"] else REJECT
            decision = "ACCEPT" if region["accept"] else "REJECT"
            self.decisions.append(label(decision, (4.8, y), color, 29, True))
            self.play(TextSwap(card, self.decisions[-1]), value_label.animate.set_color(color),
                      segment.animate.set_color(color).set_stroke(width=4.8 if region["accept"] else 2.8),
                      box.animate.set_stroke(color, opacity=0.35), run_time=0.55)
        self.wait(1.4)

    def final_result(self):
        self.play(self.stage_change("Detected segment", RESULT),
                  FadeOut(self.info), FadeOut(self.test_note),
                  *[FadeOut(item) for item in self.nfa_labels + self.decisions],
                  *[line.animate.set_opacity(0) for line in self.levels.values()],
                  self.box_a.animate.set_opacity(0), self.box_b.animate.set_opacity(0),
                  self.segment_b.animate.set_opacity(0.12), self.grid.animate.set_opacity(0.4),
                  self.caption_change("Grow locally. Validate statistically.", RESULT), run_time=0.8)
        self.wait(2.4)
