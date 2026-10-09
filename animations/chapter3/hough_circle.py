"""Hough circle intuition: a moving center, intersecting loci, radius layers.

All image and center coordinates are Cartesian, y up. The synthetic intensity
is brighter INSIDE the circle; its analytic gradient points inward. Therefore
center = point + radius * unit_gradient. Unknown polarity would require +/-.
This is exact continuous geometry, not a discretized accumulator implementation.
"""

from pathlib import Path

import numpy as np
from manim import (
    Arrow, Circle, Create, Dot, FadeIn, FadeOut, Indicate, Line,
    Scene, Succession, Text, Transform, TransformFromCopy, ValueTracker,
    VGroup, VMobject, config, linear,
)

ROOT = Path(__file__).resolve().parent
BG, WHITE, MUTED, GRID = "#0B1020", "#F8FAFC", "#8892A8", "#283248"
BLUE, YELLOW, ORANGE, PURPLE, GREEN = "#47C2FF", "#FFD166", "#FF8C42", "#B392F0", "#69D394"
CENTER = np.array([1.5, 0.5])
RADIUS = 2.0
ANGLES_DEGREES = np.array([180, 60, 300, 0, 120, 240])
LAYER_RADII = np.array([1.0, 2.0, 3.0])
IMAGE_UNIT, CENTER_UNIT = 0.56, 0.60
IMAGE_ANCHOR = np.array([-3.65, 0.25, 0.0])
CENTER_ANCHOR = np.array([4.0, 0.25, 0.0])
LAYER_ORIGIN = np.array([3.9, -1.8, 0.0])
EDGE_WIDTH = 0.15
SWEEP_START = -np.pi / 2
COLORS = [YELLOW, ORANGE, PURPLE, BLUE, BLUE, BLUE]
config.background_color = BG
config.frame_width, config.frame_height = 16, 9
config.media_dir = str(ROOT / "media")
config.text_dir = str(ROOT / "media/hough_circle/texts")


def circle_point(center, radius, phi):
    phi = np.asarray(phi)
    return np.asarray(center) + radius * np.stack((np.cos(phi), np.sin(phi)), axis=-1)


def candidate_center(point, radius, phi):
    return circle_point(point, radius, phi)


def center_locus_for_point(point, radius=RADIUS, stop=SWEEP_START + 2*np.pi):
    angles = np.linspace(SWEEP_START, stop, 241)
    return circle_point(point, radius, angles)


def synthetic_intensity(xy):
    distance = np.linalg.norm(np.asarray(xy) - CENTER, axis=-1)
    return 0.5 + 0.5 * np.tanh((RADIUS - distance) / EDGE_WIDTH)


def synthetic_gradient(xy):
    delta = np.asarray(xy) - CENTER
    distance = np.linalg.norm(delta, axis=-1)
    factor = -0.5 / EDGE_WIDTH * (1 - np.tanh((RADIUS - distance) / EDGE_WIDTH)**2)
    return factor[..., None] * delta / distance[..., None]


def compute_gradient_center_vote(point, radius, gradient):
    return np.asarray(point) + radius * np.asarray(gradient) / np.linalg.norm(gradient)


def generate_circle_example():
    points = circle_point(CENTER, RADIUS, np.deg2rad(ANGLES_DEGREES))
    gradients = synthetic_gradient(points)
    votes = np.array([compute_gradient_center_vote(p, RADIUS, g) for p, g in zip(points, gradients)])
    return dict(center=CENTER.copy(), radius=RADIUS, angles_degrees=ANGLES_DEGREES.copy(),
                points=points, gradients=gradients, gradient_votes=votes)


def image_position(xy):
    return IMAGE_ANCHOR + IMAGE_UNIT * np.r_[np.asarray(xy) - CENTER, 0.0]


def center_position(ab):
    return CENTER_ANCHOR + CENTER_UNIT * np.r_[np.asarray(ab) - CENTER, 0.0]


def layer_position(ab, radius, point):
    da, db = np.asarray(ab) - point
    # An explicit affine projection of (a,b,r), shared by all radius slices.
    return LAYER_ORIGIN + np.array([0.70*da + 0.20*db, 0.22*db + 0.95*radius, 0])


def label(text, xy, color=WHITE, size=28):
    at = np.r_[xy, 0.0] if len(xy) == 2 else xy
    return Text(text, font_size=size, color=color).move_to(at).set_z_index(8)


def path_visual(points, position, color, width=2.5):
    # Every vertex is sampled from the displayed equation; no freehand loci.
    return VMobject(color=color, stroke_width=width).set_points_as_corners([position(p) for p in points])


def locus_visual(point, radius=RADIUS, color=YELLOW):
    return path_visual(center_locus_for_point(point, radius), center_position, color)


def create_plane(position, bounds, names):
    xmin, xmax, ymin, ymax = bounds
    grid = VGroup(*[Line(position((x, ymin)), position((x, ymax)), color=GRID, stroke_width=0.55)
                    for x in range(int(np.ceil(xmin)), int(xmax)+1)],
                  *[Line(position((xmin, y)), position((xmax, y)), color=GRID, stroke_width=0.55)
                    for y in range(int(np.ceil(ymin)), int(ymax)+1)])
    axes = VGroup(Arrow(position((xmin, 0)), position((xmax, 0)), color=MUTED,
                        buff=0, stroke_width=1.2, tip_length=0.1),
                  Arrow(position((0, ymin)), position((0, ymax)), color=MUTED,
                        buff=0, stroke_width=1.2, tip_length=0.1),
                  label(names[0], position((xmax, 0)) + [0.18, -0.16, 0], MUTED, 23),
                  label(names[1], position((0, ymax)) + [-0.18, -0.17, 0], MUTED, 23))
    return VGroup(grid, axes)


class TextSwap(Succession):
    """Whole-text sequential fades, with explicit removal before replacement."""
    def __init__(self, old, new):
        super().__init__(FadeOut(old, run_time=0.2), FadeIn(new, run_time=0.25))

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


class HoughCircleTransform(Scene):
    def play(self, *animations, **kwargs):
        super().play(*animations, **kwargs)
        for root in self.mobjects:
            for mob in root.get_family():
                if mob.has_points():
                    assert np.isfinite(mob.get_all_points()).all()
                    assert mob.get_left()[0] > -7.95 and mob.get_right()[0] < 7.95
                    assert mob.get_bottom()[1] > -4.45 and mob.get_top()[1] < 4.45

    def swap(self, name, new):
        old = getattr(self, name)
        setattr(self, name, new)
        return TextSwap(old, new)

    def cue(self, text):
        new = label(text, (0, -3.45), size=29)
        assert new.get_top()[1] < -3.12 and new.get_bottom()[1] > -3.8
        return self.swap("caption", new)

    def progress(self, index):
        return self.marker.animate.move_to([self.steps[index].get_x(), -4.26, 0])

    def construct(self):
        self.data = generate_circle_example()
        self.points = self.data["points"]
        self.caption = label("Where is the center?", (0, -3.45), size=29)
        self.steps = VGroup(*[label(name, (x, -4.03), MUTED, 22)
                             for x, name in zip(np.linspace(-6.1, 6.1, 6),
                                                ["Edge", "Centers", "Intersect", "Radius", "Gradient", "Circle"])])
        self.marker = Dot([-6.1, -4.26, 0], radius=0.045, color=BLUE)
        self.add(label("Hough circles · where do the votes meet?", (0, 3.83), size=35),
                 self.caption, self.steps, self.marker)
        self.show_image()
        self.moving_candidate()
        self.trace_center_locus()
        self.intersect_loci()
        self.add_more_points()
        self.unknown_radius()
        self.gradient_acceleration()
        self.finish_circle()
        assert not any(mob.updaters for root in self.mobjects for mob in root.get_family())

    def show_image(self):
        self.image_plane = create_plane(image_position, (-5, 6, -4, 5), ("x", "y"))
        self.image_title = label("Image Space", (-3.65, 3.05), size=30)
        self.circle = Circle(radius=IMAGE_UNIT*RADIUS, color=BLUE, stroke_width=3)
        self.circle.move_to(image_position(CENTER))
        self.edge_dots = VGroup(*[Dot(image_position(p), radius=0.07, color=BLUE).set_z_index(6)
                                 for p in self.points])
        self.point_labels = [label(f"p{i+1}", image_position(self.points[i]) + offset, COLORS[i], 26)
                             for i, offset in enumerate(([-0.34, 0.28, 0], [0.27, 0.28, 0], [0.30, -0.27, 0]))]
        self.play(FadeIn(self.image_plane), FadeIn(self.image_title), run_time=0.6)
        self.play(Create(self.circle), run_time=0.85)
        self.play(*[FadeIn(dot) for dot in self.edge_dots[:3]], run_time=0.5)
        self.wait(0.9)

    def moving_candidate(self):
        point = self.points[0]
        self.angle = ValueTracker(SWEEP_START)
        self.left_note = label("Known radius: r = 2", (-3.65, -2.8), MUTED, 28)
        self.play(self.edge_dots[0].animate.set_color(YELLOW), FadeIn(self.point_labels[0]),
                  FadeIn(self.left_note), self.cue("One point. A fixed radius. Many possible centers."), run_time=0.5)
        self.candidate_dot = Dot(image_position(candidate_center(point, RADIUS, self.angle.get_value())),
                                 color=YELLOW, radius=0.065).set_z_index(6)
        self.candidate_circle = Circle(radius=IMAGE_UNIT*RADIUS, color=YELLOW, stroke_width=1.8,
                                        stroke_opacity=0.55).move_to(self.candidate_dot)
        self.compass = Line(image_position(point), self.candidate_dot.get_center(), color=MUTED, stroke_width=2)
        self.r_tag = label("r", self.compass.get_center() + [0.16, 0.1, 0], MUTED, 26)
        self.candidate_dot.add_updater(lambda m: m.move_to(image_position(candidate_center(point, RADIUS, self.angle.get_value()))))
        self.candidate_circle.add_updater(lambda m: m.move_to(self.candidate_dot))
        self.compass.add_updater(lambda m: m.put_start_and_end_on(image_position(point), self.candidate_dot.get_center()))
        self.r_tag.add_updater(lambda m: m.move_to(self.compass.get_center() + [0.16, 0.1, 0]))
        self.play(FadeIn(self.candidate_dot), Create(self.candidate_circle), Create(self.compass), FadeIn(self.r_tag), run_time=0.5)
        self.play(self.angle.animate.set_value(SWEEP_START + 2*np.pi), run_time=2.0, rate_func=linear)
        self.wait(0.5)

    def trace_center_locus(self):
        point = self.points[0]
        self.center_plane = create_plane(center_position, (-3, 6, -3.8, 4.6), ("a", "b"))
        self.center_title = label("Center Space (a, b)", (4.0, 3.05), size=30)
        self.right_note = label("(x1 − a)² + (y1 − b)² = r²", (4.0, -2.8), size=27)
        self.play(FadeIn(self.center_plane), FadeIn(self.center_title), self.progress(1), run_time=0.55)
        self.center_dot = Dot(center_position(candidate_center(point, RADIUS, SWEEP_START)), color=YELLOW, radius=0.07)
        self.play(TransformFromCopy(self.candidate_dot, self.center_dot),
                  self.cue("One edge point → a circle of possible centers"), run_time=0.55)
        self.angle.set_value(SWEEP_START)
        first = path_visual(center_locus_for_point(point, stop=SWEEP_START+1e-6), center_position, YELLOW)
        first.add_updater(lambda m: m.become(path_visual(center_locus_for_point(
            point, stop=max(SWEEP_START+1e-6, self.angle.get_value())), center_position, YELLOW)))
        self.center_dot.add_updater(lambda m: m.move_to(center_position(candidate_center(point, RADIUS, self.angle.get_value()))))
        self.add(first)
        self.play(self.angle.animate.set_value(SWEEP_START + 2*np.pi), run_time=2.7, rate_func=linear)
        first.clear_updaters()
        first.become(locus_visual(point))
        for mob in (self.candidate_dot, self.candidate_circle, self.compass, self.r_tag, self.center_dot):
            mob.clear_updaters()
        self.loci = [first]
        self.play(FadeIn(self.right_note), run_time=0.4)
        self.wait(0.7)

    def intersect_loci(self):
        self.play(*[FadeOut(m) for m in (self.candidate_dot, self.candidate_circle, self.compass, self.r_tag, self.center_dot)],
                  self.cue("Other edge points vote for the same center"), self.progress(2), run_time=0.45)
        for i in (1, 2):
            locus = locus_visual(self.points[i], color=COLORS[i])
            self.play(self.edge_dots[i].animate.set_color(COLORS[i]), FadeIn(self.point_labels[i]),
                      Create(locus), run_time=1.0)
            self.loci.append(locus)
        self.true_center = Dot(image_position(CENTER), radius=0.085, color=GREEN).set_z_index(7)
        self.peak = Dot(center_position(CENTER), radius=0.10, color=GREEN).set_z_index(7)
        self.peak_ring = Circle(radius=0.19, color=GREEN, stroke_width=2).move_to(self.peak)
        self.center_tag = label("(a*, b*)", image_position(CENTER) + [0, -0.43, 0], GREEN, 25)
        self.peak_tag = label("(a*, b*)", center_position(CENTER) + [0.9, -0.4, 0], GREEN, 26)
        self.play(FadeIn(self.true_center), FadeIn(self.peak), Create(self.peak_ring),
                  FadeIn(self.center_tag), FadeIn(self.peak_tag),
                  self.swap("right_note", label("(a*, b*) = (1.5, 0.5)", (4.0, -2.8), GREEN, 28)),
                  self.cue("One intersection ↔ the real circle center"), run_time=0.65)
        self.wait(1.65)

    def add_more_points(self):
        self.play(*[locus.animate.set_stroke(opacity=0.45) for locus in self.loci], run_time=0.35)
        for i in (3, 4, 5):
            locus = locus_visual(self.points[i], color=BLUE).set_stroke(opacity=0.40)
            self.play(FadeIn(self.edge_dots[i]), Create(locus), run_time=0.3)
            self.loci.append(locus)
        self.play(Indicate(self.peak_ring, color=GREEN, scale_factor=1.4),
                  self.cue("Many edge pixels → one center peak"), run_time=0.5)
        self.wait(0.75)

    def unknown_radius(self):
        point = self.points[0]
        self.play(self.cue("What if the radius is unknown?"), self.progress(3),
                  *[locus.animate.set_stroke(opacity=0.08) for locus in self.loci[1:]],
                  FadeOut(self.peak), FadeOut(self.peak_ring), FadeOut(self.peak_tag),
                  self.loci[0].animate.set_stroke(opacity=1), run_time=0.5)
        self.play(Transform(self.loci[0], locus_visual(point, 1.0)),
                  self.swap("right_note", label("r = 1", (4, -2.8), YELLOW, 29)), run_time=0.6)
        self.play(Transform(self.loci[0], locus_visual(point, 3.0)),
                  self.swap("right_note", label("r = 3", (4, -2.8), YELLOW, 29)), run_time=0.6)
        # Actual radius slices project into a single shared (a,b,r) frame.
        self.layers = [path_visual(center_locus_for_point(point, r),
                                   lambda ab, r=r: layer_position(ab, r, point), YELLOW)
                       for r in LAYER_RADII]
        self.play(Transform(self.loci[0], self.layers[1]), self.center_plane.animate.set_opacity(0.10),
                  *[locus.animate.set_opacity(0) for locus in self.loci[1:]],
                  self.swap("center_title", label("Center + radius (a, b, r)", (4, 3.05), size=29)),
                  self.swap("left_note", label("Line: 2 parameters (ρ, θ)", (-3.65, -2.8), MUTED, 26)),
                  self.swap("right_note", label("Circle: 3 parameters (a, b, r)", (4, -2.8), YELLOW, 26)),
                  run_time=0.8)
        self.layer_helpers = VGroup(
            Arrow([6.6, -1.9, 0], [6.6, 1.65, 0], buff=0, color=MUTED, stroke_width=1.8),
            label("r", (6.6, 1.95), MUTED, 27),
            *[label(f"r = {r:.0f}", (7.15, -1.8 + 0.95*r), YELLOW, 24) for r in LAYER_RADII],
            *[Line(LAYER_ORIGIN, layer_position(circle_point(point, 3, phi), 3, point),
                   color=MUTED, stroke_width=1.2) for phi in (0, np.pi)],
            Dot(LAYER_ORIGIN, radius=0.04, color=MUTED))
        self.accumulator = label("A[a, b, r]", (4, 2.18), YELLOW, 31)
        self.play(TransformFromCopy(self.loci[0], self.layers[0]),
                  TransformFromCopy(self.loci[0], self.layers[2]), FadeIn(self.layer_helpers),
                  FadeIn(self.accumulator), self.cue("Stack radius slices → a cone of possible centers"), run_time=0.7)
        self.wait(0.7)
        self.play(self.cue("An extra dimension means more storage and votes"), run_time=0.5)
        self.wait(1.5)

    def gradient_acceleration(self):
        point, gradient = self.points[0], self.data["gradients"][0]
        unit_gradient = gradient / np.linalg.norm(gradient)
        self.play(FadeOut(self.layers[0]), FadeOut(self.layers[2]), FadeOut(self.layer_helpers),
                  FadeOut(self.accumulator), Transform(self.loci[0], locus_visual(point)),
                  self.center_plane.animate.set_opacity(1), self.progress(4),
                  self.swap("center_title", label("Center Space (a, b)", (4, 3.05), size=30)),
                  self.swap("left_note", label("Bright inside → inward gradient", (-3.65, -2.8), YELLOW, 25)),
                  self.swap("right_note", label("Fixed r = 2", (4, -2.8), MUTED, 28)),
                  self.cue("The gradient tells us which way the center lies"), run_time=0.7)
        self.gradient_arrow = Arrow(image_position(point), image_position(point + 1.5*unit_gradient),
                                    buff=0, color=YELLOW, stroke_width=4, tip_length=0.15)
        self.gradient_tag = label("gradient", image_position(point + 0.75*unit_gradient) + [0, 0.70, 0], YELLOW, 27)
        self.radial_line = Line(image_position(point), image_position(CENTER), color=GREEN, stroke_width=2)
        self.play(self.circle.animate.set_fill(BLUE, opacity=0.10), Create(self.radial_line),
                  Create(self.gradient_arrow), FadeIn(self.gradient_tag), run_time=0.6)
        self.wait(1.05)
        self.vote_ray = Arrow(center_position(point), center_position(point + 3.0*unit_gradient),
                              buff=0, color=GREEN, stroke_width=3, tip_length=0.13)
        votes = circle_point(point, RADIUS, np.linspace(0, 2*np.pi, 24, endpoint=False))
        self.vote_dots = VGroup(*[Dot(center_position(p), radius=0.042, color=YELLOW) for p in votes])
        self.play(FadeIn(self.vote_dots), Create(self.vote_ray),
                  self.swap("right_note", label("center = p + r · unit gradient", (4, -2.8), GREEN, 26)), run_time=0.55)
        self.wait(0.4)
        # Illustrates removal of directional ambiguity, NOT transfer of actual
        # accumulator counts: all 24 dots are alternative centers for ONE point.
        predicted = self.data["gradient_votes"][0]
        self.play(*[dot.animate.move_to(center_position(predicted)).set_color(GREEN) for dot in self.vote_dots],
                  self.loci[0].animate.set_stroke(opacity=0.18),
                  self.cue("Known inward direction → one center per radius"), run_time=1.0)
        self.play(FadeOut(self.vote_dots), FadeIn(self.peak), FadeIn(self.peak_ring),
                  self.cue("Gradient direction reduces the search"), run_time=0.5)
        self.wait(0.9)

    def finish_circle(self):
        self.play(FadeOut(self.vote_ray), FadeOut(self.gradient_arrow), FadeOut(self.gradient_tag),
                  FadeOut(self.radial_line),
                  *[locus.animate.set_stroke(opacity=0.26) for locus in self.loci],
                  self.circle.animate.set_color(GREEN).set_stroke(width=3.5),
                  FadeIn(self.peak_tag), self.progress(5),
                  self.swap("left_note", label("(a, b) = (1.5, 0.5) · r = 2", (-3.65, -2.8), GREEN, 27)),
                  self.swap("right_note", label("Peak at (a, b, r)", (4, -2.8), GREEN, 29)),
                  self.cue("Edge pixels vote for center + radius"), run_time=0.8)
        self.wait(2.2)
