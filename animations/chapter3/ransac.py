"""One persistent point cloud: sample, perpendicular consensus, and TLS refit.

The first three draws of SAMPLE_SEED are shown in order (no hidden iterations).
Seeds were chosen for a clear bad / intermediate / good teaching example.
All geometry and counts derive from the same numerical models and epsilon.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import (
    Arrow, Circle, Create, DashedLine, Dot, FadeIn, FadeOut, Indicate,
    LaggedStart, Line, Polygon, Scene, Succession, Text, Transform, TransformMatchingShapes,
    VGroup, config,
)


ROOT = Path(__file__).resolve().parent
BACKGROUND, DATA, ACTIVE = "#0B1020", "#47C2FF", "#FFD166"
RESULT, REJECT, MUTED = "#69D394", "#FF8C42", "#8892A8"
NEUTRAL, GRID, WHITE = "#59849D", "#283248", "#F8FAFC"
config.background_color = BACKGROUND
config.frame_width, config.frame_height = 16, 9
config.media_dir = str(ROOT / "media")
config.text_dir = str(ROOT / "media/ransac/texts")

DATA_SEED, SAMPLE_SEED = 37, 7647
EPSILON = 0.22
TRUE_SLOPE, TRUE_INTERCEPT = 0.65, 0.3
BOUNDS = (-4.5, 4.5, -3.2, 3.2)
ORIGIN = np.array([-2.4, 0.0, 0.0])
UNIT = 0.85  # Equal x/y scale preserves perpendicular distances on screen.


def generate_dataset():
    rng = np.random.default_rng(DATA_SEED)
    x = np.linspace(-3.4, 3.4, 14)
    inliers = np.column_stack((x, TRUE_SLOPE * x + TRUE_INTERCEPT
                              + rng.normal(0, 0.095, len(x))))
    outliers = np.array([[-3.8, 2.6], [-2.5, 2.5], [-0.7, -2.6],
                         [1.0, -2.3], [3.3, -2.7], [4.0, -2.4]])
    return np.vstack((inliers, outliers)), np.arange(20) < 14


def normalize_line(coefficients):
    model = np.asarray(coefficients, dtype=float).copy()
    length = np.linalg.norm(model[:2])
    if not np.isfinite(model).all() or length < 1e-12:
        raise ValueError("A finite non-degenerate line is required")
    model /= length
    # Canonical sign, with b positive (or a positive for vertical lines).
    if model[1] < -1e-12 or (abs(model[1]) <= 1e-12 and model[0] < 0):
        model *= -1
    return model


def line_from_two_points(p1, p2):
    delta = np.asarray(p2) - p1
    normal = np.array([-delta[1], delta[0]])
    return normalize_line([*normal, -np.dot(normal, p1)])


def point_line_distance(points, model):
    model = normalize_line(model)
    return np.abs(np.asarray(points) @ model[:2] + model[2])


def classify_inliers(points, model, epsilon=EPSILON):
    return point_line_distance(points, model) <= epsilon


def fit_line_to_consensus(points):
    """Total least squares: minimize squared perpendicular residuals."""
    center = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(np.asarray(points) - center, full_matrices=False)
    normal = vh[-1]
    return normalize_line([*normal, -np.dot(normal, center)])


def least_squares(points):
    slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
    return normalize_line([-slope, 1, -intercept])


def experiment():
    points, intended = generate_dataset()
    rng = np.random.default_rng(SAMPLE_SEED)
    pairs = [rng.choice(len(points), size=2, replace=False) for _ in range(3)]
    models = [line_from_two_points(*points[pair]) for pair in pairs]
    masks = [classify_inliers(points, model) for model in models]
    best = int(np.argmax([mask.sum() for mask in masks]))
    final = fit_line_to_consensus(points[masks[best]])
    return points, intended, pairs, models, masks, best, final


def position(point):
    return ORIGIN + UNIT * np.array([point[0], point[1], 0])


def line_endpoints(model):
    """Clip any non-degenerate infinite line to the plot, including verticals."""
    model = normalize_line(model)
    foot = -model[2] * model[:2]
    tangent = np.array([model[1], -model[0]])
    lo, hi = -np.inf, np.inf
    for axis, low, high in ((0, BOUNDS[0], BOUNDS[1]), (1, BOUNDS[2], BOUNDS[3])):
        if abs(tangent[axis]) < 1e-12:
            if not low <= foot[axis] <= high:
                raise ValueError("Line misses the plot")
        else:
            ends = sorted(((low - foot[axis]) / tangent[axis],
                           (high - foot[axis]) / tangent[axis]))
            lo, hi = max(lo, ends[0]), min(hi, ends[1])
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("Line misses the plot")
    return np.array([foot + lo * tangent, foot + hi * tangent])


def band_vertices(model, epsilon=EPSILON):
    """Clip the exact |a*x+b*y+c| <= epsilon strip to the plot rectangle."""
    model = normalize_line(model)
    xmin, xmax, ymin, ymax = BOUNDS
    vertices = [np.array(p, dtype=float) for p in
                ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))]
    for sign in (-1, 1):
        output = []
        for p, q in zip(vertices, vertices[1:] + vertices[:1]):
            dp = sign * (p @ model[:2] + model[2]) - epsilon
            dq = sign * (q @ model[:2] + model[2]) - epsilon
            if dp <= 1e-12:
                output.append(p)
            if (dp <= 0) != (dq <= 0):
                output.append(p + (q - p) * dp / (dp - dq))
        vertices = output
    return np.asarray(vertices)


def line_to_manim_object(model, color=REJECT, dashed=False):
    p, q = map(position, line_endpoints(model))
    cls = DashedLine if dashed else Line
    return cls(p, q, color=color, stroke_width=3.2).set_z_index(3)


def create_epsilon_band(model, opacity=0.18):
    return Polygon(*map(position, band_vertices(model)), stroke_width=0,
                   fill_color=RESULT, fill_opacity=opacity).set_z_index(1)


def label(text, xy, color=WHITE, size=27, bold=False):
    return Text(text, font_size=size, color=color,
                weight="BOLD" if bold else "NORMAL").move_to([*xy, 0])


def create_plane():
    xmin, xmax, ymin, ymax = BOUNDS
    grid = VGroup(*[Line(position((x, ymin)), position((x, ymax)),
                        color=GRID, stroke_width=0.7) for x in range(-4, 5)],
                  *[Line(position((xmin, y)), position((xmax, y)),
                         color=GRID, stroke_width=0.7) for y in range(-3, 4)])
    axes = VGroup(Arrow(position((xmin, 0)), position((xmax + 0.17, 0)),
                        color=MUTED, buff=0, stroke_width=1.4, tip_length=0.1),
                  Arrow(position((0, ymin)), position((0, ymax + 0.17)),
                        color=MUTED, buff=0, stroke_width=1.4, tip_length=0.1))
    axes.add(label("x", position((xmax + 0.38, 0))[:2], MUTED, 23),
             label("y", position((0, ymax + 0.45))[:2], MUTED, 23))
    for x in (-4, -2, 2, 4):
        axes.add(label(str(x), position((x, -0.27))[:2], MUTED, 20))
    for y in (-2, 2):
        axes.add(label(str(y), position((-0.27, y))[:2], MUTED, 20))
    return VGroup(grid, axes)


class CaptionTransition(Succession):
    """Fade whole sentences sequentially within the existing play duration."""

    def __init__(self, old, new):
        super().__init__(FadeOut(old, run_time=0.20), FadeIn(new, run_time=0.25))

    def next_animation(self):
        if self.active_index == 0:
            self.active_animation.finish()
            # Succession normally defers FadeOut cleanup until the whole play
            # ends. Remove the old sentence BEFORE FadeIn sets up the new one.
            self.scene.remove(self.active_animation.mobject)
            self.update_active_animation(1)
        else:
            super().next_animation()

    def interpolate(self, alpha):
        # Longer concurrent geometry animations hold the new sentence after
        # 0.45 s; the existing 0.40 s play scales the fades to 0.178/0.222 s.
        # No play/wait duration or non-text animation is changed.
        progress = alpha * max(self.run_time, self.max_end_time) / self.max_end_time
        super().interpolate(min(progress, 1.0))


class RANSACLineFitting(Scene):
    """Three seeded hypotheses on the same twenty living point objects."""

    def play(self, *animations, **kwargs):
        super().play(*animations, **kwargs)
        for root in self.mobjects:
            for mob in root.get_family():
                if isinstance(mob, Text) and mob.width and mob.height:
                    assert mob.get_left()[0] > -7.9 and mob.get_right()[0] < 7.9, mob.text
                    assert mob.get_bottom()[1] > -4.4 and mob.get_top()[1] < 4.4, mob.text

    def morph(self, name, target):
        source = getattr(self, name)
        setattr(self, name, target)
        return TransformMatchingShapes(source, target)

    def caption_change(self, text, color=WHITE):
        old = self.caption
        new = label(text, (0, -3.38), color, 28)
        # Dedicated caption strip: below the plot/panel and above progress.
        assert new.get_bottom()[1] > -3.7 and new.get_top()[1] < -3.05
        self.caption = new
        return CaptionTransition(old, new)

    def step(self, active):
        return [text.animate.set_color(ACTIVE if i == active else MUTED)
                for i, text in enumerate(self.progress)]

    def construct(self):
        (self.points, self.intended, self.pairs, self.models,
         self.masks, self.best, self.final) = experiment()
        self.dots = VGroup(*[Dot(position(p), radius=0.082, color=NEUTRAL).set_z_index(6)
                            for p in self.points])
        self.caption = label("Which line describes the dominant structure?", (0, -3.38), size=28)
        self.add(label("RANSAC", (0, 3.88), size=42, bold=True))
        self.problem_setup()
        self.introduce_ransac()
        self.iteration(0)
        self.iteration(1)
        self.iteration(2)
        self.refit_and_compare()
        assert all(not mob.updaters for root in self.mobjects for mob in root.get_family())

    def problem_setup(self):
        self.panel_title = label("Least Squares", (4.6, 2.5), size=30, bold=True)
        self.sample_note = label("Fit all visible points", (4.6, 1.96), MUTED, 23)
        self.play(Create(create_plane()), FadeIn(self.dots[:18]), FadeIn(self.caption),
                  FadeIn(self.panel_title), FadeIn(self.sample_note), run_time=0.8)
        self.wait(0.6)
        self.ls = line_to_manim_object(least_squares(self.points[:18]), REJECT)
        self.play(Create(self.ls), run_time=0.65)
        self.wait(0.4)
        self.dots[18:].set_color(REJECT)
        self.play(LaggedStart(*[FadeIn(dot, scale=0.3) for dot in self.dots[18:]], lag_ratio=0.2),
                  Transform(self.ls, line_to_manim_object(least_squares(self.points), REJECT)),
                  self.caption_change("Outliers pull the fit", REJECT), run_time=1.2)
        self.wait(1.3)

    def introduce_ransac(self):
        self.progress = VGroup(*[label(text, (x, -4.02), MUTED, 23)
                                for text, x in zip(("1. Sample", "2. Fit", "3. Count inliers",
                                                    "4. Repeat", "5. Refit"),
                                                   (-5.65, -3.0, 0, 3.1, 5.75))])
        self.score_title = label("Consensus", (4.6, 1.12), MUTED, 27)
        self.score = label("— / 20", (4.6, 0.53), size=40, bold=True)
        self.epsilon_note = label("ε band: 0.22", (4.6, -0.22), RESULT, 26)
        self.distance_note = label("Perpendicular distance", (4.6, -0.74), MUTED, 23)
        self.history = VGroup()
        self.play(FadeIn(self.progress), self.ls.animate.set_opacity(0),
                  self.morph("panel_title", label("Iteration 1", (4.6, 2.5), size=30, bold=True)),
                  self.morph("sample_note", label("Minimum sample: 2 points", (4.6, 1.96), ACTIVE, 23)),
                  self.caption_change("Try a line from just two points"),
                  *[dot.animate.set_color(NEUTRAL) for dot in self.dots], run_time=0.65)
        self.remove(self.ls)
        self.wait(0.55)

    def residual_geometry(self, model, indices):
        geometry = VGroup()
        tangent = np.array([model[1], -model[0]])
        for i in indices:
            point = self.points[i]
            signed = point @ model[:2] + model[2]
            foot = point - signed * model[:2]
            if abs(signed) < 1e-8:
                continue
            geometry.add(DashedLine(position(point), position(foot), color=MUTED,
                                    stroke_width=1.4, dash_length=0.065))
            normal = np.sign(signed) * model[:2]
            corner = [foot + tangent * 0.13,
                      foot + tangent * 0.13 + normal * 0.13,
                      foot + normal * 0.13]
            geometry.add(Line(position(corner[0]), position(corner[1]), color=MUTED, stroke_width=1.2),
                         Line(position(corner[1]), position(corner[2]), color=MUTED, stroke_width=1.2))
        return geometry.set_z_index(4)

    def iteration(self, index):
        pair, model, mask = self.pairs[index], self.models[index], self.masks[index]
        good = index == self.best
        color = RESULT if good else REJECT
        if index:
            self.play(*self.step(3),
                      self.caption_change("New sample → new hypothesis"),
                      self.band.animate.set_fill(opacity=0.035),
                      self.candidate.animate.set_stroke(opacity=0.3),
                      *[dot.animate.set_color(NEUTRAL) for dot in self.dots],
                      self.morph("score", label("— / 20", (4.6, 0.53), size=40, bold=True)),
                      run_time=0.45)
            self.play(self.morph("panel_title", label(f"Iteration {index + 1}", (4.6, 2.5), size=30, bold=True)),
                      run_time=0.25)
        rings = VGroup(*[Circle(radius=0.15, color=ACTIVE, stroke_width=2).move_to(self.dots[i])
                         for i in pair]).set_z_index(7)
        self.play(*self.step(0), Create(rings),
                  *[self.dots[i].animate.set_color(ACTIVE) for i in pair],
                  self.caption_change("Random pair → candidate line", ACTIVE),
                  run_time=0.65 if index != 1 else 0.4)
        self.wait(0.35 if index != 1 else 0.15)
        target = line_to_manim_object(model, color)
        if index == 0:
            self.candidate = target
            self.play(Create(self.candidate), *self.step(1), run_time=0.75)
        else:
            self.play(Transform(self.candidate, target), *self.step(1),
                      run_time=0.75 if good else 0.5)
        band = create_epsilon_band(model)
        if index == 0:
            self.band = band
            self.play(FadeIn(self.band), FadeIn(self.epsilon_note), FadeIn(self.distance_note),
                      self.caption_change("Measure perpendicular distance to the line"), run_time=0.65)
            helpers = self.residual_geometry(model, [4, 7, 12])
            self.play(Create(helpers), run_time=0.7)
            self.wait(1.15)
            self.play(FadeOut(helpers), self.morph("distance_note", label("d ≤ ε → inlier", (4.6, -0.74), DATA, 25)),
                      run_time=0.35)
        else:
            self.play(Transform(self.band, band), run_time=0.45 if good else 0.3)
        self.play(*self.step(2), FadeOut(rings),
                  LaggedStart(*[dot.animate.set_color(DATA if keep else REJECT)
                                for dot, keep in zip(self.dots, mask)], lag_ratio=0.025),
                  *([FadeIn(self.score_title), FadeIn(self.score)] if index == 0 else []),
                  run_time=0.95 if index != 1 else 0.5)
        count = int(mask.sum())
        row = label(f"Trial {index + 1}:  {count} / 20", (4.6, -1.55 - 0.48 * index),
                    RESULT if good else MUTED, 25, good)
        self.history.add(row)
        message = ("Poor model: little support", "Better, but incomplete", "A good sample reveals the structure")[index]
        self.play(self.morph("score", label(f"{count} / 20", (4.6, 0.53), color, 40, True)),
                  FadeIn(row), self.caption_change(message, color), run_time=0.55)
        self.wait(1.85 if good else (0.9 if index == 0 else 0.55))

    def refit_and_compare(self):
        mask = self.masks[self.best]
        self.play(Indicate(self.history[self.best], color=RESULT, scale_factor=1.06),
                  self.caption_change("Keep the largest consensus", RESULT), run_time=0.6)
        self.wait(0.7)
        halo = VGroup(*[Circle(radius=0.14, stroke_width=1.6, color=DATA).move_to(dot)
                        for dot, keep in zip(self.dots, mask) if keep]).set_z_index(7)
        self.play(Create(halo), *self.step(4),
                  self.morph("sample_note", label(f"Refit: all {int(mask.sum())} inliers", (4.6, 1.96), DATA, 25)),
                  self.morph("panel_title", label("Best consensus", (4.6, 2.5), RESULT, 30, True)),
                  self.caption_change("Refit using all inliers", RESULT), run_time=0.8)
        self.wait(0.65)
        self.play(Transform(self.candidate, line_to_manim_object(self.final, RESULT).set_stroke(width=4.5)),
                  Transform(self.band, create_epsilon_band(self.final, opacity=0.07)),
                  FadeOut(halo), run_time=1.25)
        self.wait(2.0)
        comparison = line_to_manim_object(least_squares(self.points), REJECT, dashed=True).set_opacity(0.5)
        legend = VGroup(label("Dashed: LS on all points", (4.6, -0.22), REJECT, 23),
                        label("Green: consensus refit", (4.6, -0.74), RESULT, 23))
        self.play(Create(comparison), Transform(self.epsilon_note, legend[0]),
                  Transform(self.distance_note, legend[1]),
                  self.caption_change("Outliers pull LS; consensus preserves structure"), run_time=0.8)
        self.wait(1.55)
        self.play(self.caption_change("Sample → Consensus → Refit", RESULT), run_time=0.55)
        self.wait(1.8)
