"""One continuous explanation: image points become Hough sinusoids.

Chapter 3 convention: rho = x*cos(theta) + y*sin(theta), theta in [0, pi).
Image coordinates here are Cartesian (y up); rho is a signed distance.
All displayed curves, normals, candidate lines and markers share that model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import (
    Arc, Arrow, Circle, Create, DecimalNumber, Dot, FadeIn, Indicate,
    LaggedStart, Line, Scene, Text, Transform, TransformMatchingShapes, ValueTracker, VGroup,
    VMobject, config, linear,
)


ROOT = Path(__file__).resolve().parent
BACKGROUND = "#0B1020"
DATA = "#47C2FF"
ACTIVE = "#FFD166"
SECOND = "#FF8C42"
RESULT = "#69D394"
MUTED = "#8892A8"
GRID = "#283248"
WHITE = "#F8FAFC"

# Keep render paths local even when Manim is invoked from the repository root.
# Text cache is scene-specific so cleanup never touches Canny's artifacts.
config.background_color = BACKGROUND
config.frame_width = 16
config.frame_height = 9
config.media_dir = str(ROOT / "media")
config.text_dir = str(ROOT / "media/hough_transform/texts")

P1 = np.array([1.0, 2.0])
P2 = np.array([3.0, 1.0])
EXTRA_POINTS = np.array([[0.2, 2.4], [1.8, 1.6], [3.7, 0.65], [4.5, 0.25]])
THETA_END = np.deg2rad(179.9)
IMAGE_BOUNDS = (-1.0, 5.0, -1.3, 4.0)
IMAGE_ORIGIN = np.array([-5.55, -1.30, 0.0])
IMAGE_UNIT = 0.85
HOUGH_ORIGIN = np.array([1.10, -0.15, 0.0])
HOUGH_WIDTH = 5.5
RHO_UNIT = 0.45


def compute_rho(point: np.ndarray, theta: float | np.ndarray):
    """Signed projection of an image-space point on the unit normal."""
    return point[0] * np.cos(theta) + point[1] * np.sin(theta)


def common_line_parameters(p1=P1, p2=P2) -> tuple[float, float]:
    tangent = np.asarray(p2) - np.asarray(p1)
    normal = np.array([-tangent[1], tangent[0]])
    if np.linalg.norm(normal) < 1e-12:
        raise ValueError("Two distinct points are required")
    theta = float(np.arctan2(normal[1], normal[0]) % np.pi)
    return float(compute_rho(p1, theta)), theta


def image_position(point: np.ndarray | tuple) -> np.ndarray:
    return IMAGE_ORIGIN + IMAGE_UNIT * np.array([point[0], point[1], 0.0])


def parameter_position(theta: float, rho: float) -> np.ndarray:
    return HOUGH_ORIGIN + np.array([HOUGH_WIDTH * theta / np.pi, RHO_UNIT * rho, 0.0])


def candidate_endpoints(point: np.ndarray, theta: float) -> np.ndarray:
    """Clip an infinite line through point to the image-plane rectangle.

    Parameterizing by its tangent avoids slope singularities at vertical lines.
    """
    tangent = np.array([-np.sin(theta), np.cos(theta)])
    t_low, t_high = -np.inf, np.inf
    xmin, xmax, ymin, ymax = IMAGE_BOUNDS
    for axis, low, high in ((0, xmin, xmax), (1, ymin, ymax)):
        if abs(tangent[axis]) > 1e-12:
            limits = sorted(((low - point[axis]) / tangent[axis],
                             (high - point[axis]) / tangent[axis]))
            t_low, t_high = max(t_low, limits[0]), min(t_high, limits[1])
    if not np.isfinite([t_low, t_high]).all() or t_low > t_high:
        raise ValueError("Candidate line does not intersect the image plane")
    return np.array([point + t_low * tangent, point + t_high * tangent])


def hough_curve_points(point: np.ndarray, stop: float = THETA_END,
                       samples: int = 361) -> np.ndarray:
    """Numerically sample (theta, rho); include the exact moving endpoint."""
    angles = np.linspace(0.0, stop, max(2, samples))
    return np.column_stack((angles, compute_rho(point, angles)))


def label(text: str, position, color=WHITE, size=24, bold=False) -> Text:
    # Match the approved Canny scene's Pango/default-font typography.
    return Text(text, font_size=size, color=color,
                weight="BOLD" if bold else "NORMAL").move_to(position)


def create_image_plane() -> VGroup:
    xmin, xmax, ymin, ymax = IMAGE_BOUNDS
    grid = VGroup()
    for x in range(-1, 6):
        grid.add(Line(image_position((x, ymin)), image_position((x, ymax)),
                      color=GRID, stroke_width=0.8))
    for y in range(-1, 5):
        grid.add(Line(image_position((xmin, y)), image_position((xmax, y)),
                      color=GRID, stroke_width=0.8))
    axes = VGroup(
        Arrow(image_position((xmin, 0)), image_position((5.18, 0)),
              buff=0, color=MUTED, stroke_width=1.7, tip_length=0.11),
        Arrow(image_position((0, ymin)), image_position((0, 4.18)),
              buff=0, color=MUTED, stroke_width=1.7, tip_length=0.11),
        label("x", image_position((5.38, 0)), MUTED),
        label("y", image_position((0, 4.43)), MUTED),
        label("0", image_position((-0.22, -0.23)), MUTED, 21),
    )
    for x in (2, 4):
        axes.add(label(str(x), image_position((x, -0.25)), MUTED, 21))
    for y in (2, 4):
        axes.add(label(str(y), image_position((-0.25, y)), MUTED, 21))
    return VGroup(grid, axes)


def create_hough_plane() -> VGroup:
    grid = VGroup()
    for theta in np.linspace(0, np.pi, 5):
        grid.add(Line(parameter_position(theta, -5), parameter_position(theta, 5),
                      color=GRID, stroke_width=0.8))
    for rho in (-4, -2, 0, 2, 4):
        grid.add(Line(parameter_position(0, rho), parameter_position(np.pi, rho),
                      color=MUTED if rho == 0 else GRID,
                      stroke_width=1.3 if rho == 0 else 0.8))
    axes = VGroup(
        Arrow(parameter_position(0, -5), parameter_position(0, 5.25),
              buff=0, color=MUTED, stroke_width=1.7, tip_length=0.11),
        Arrow(parameter_position(0, -5), parameter_position(np.pi + 0.08, -5),
              buff=0, color=MUTED, stroke_width=1.7, tip_length=0.11),
        label("ρ", parameter_position(0, 5.6), MUTED),
        label("θ", parameter_position(np.pi + 0.25, -5), MUTED),
    )
    for theta, text in ((0, "0"), (np.pi / 2, "π/2"), (np.pi, "π")):
        axes.add(label(text, parameter_position(theta, -5.65), MUTED, 22))
    for rho in (-4, -2, 0, 2, 4):
        axes.add(label(str(rho), parameter_position(-0.20, rho), MUTED, 21))
    return VGroup(grid, axes)


def create_candidate_line(point: np.ndarray, theta: float, color=DATA) -> Line:
    a, b = candidate_endpoints(point, theta)
    return Line(image_position(a), image_position(b), color=color,
                stroke_width=3).set_z_index(2)


def create_curve(point, stop=THETA_END, color=DATA, opacity=1.0) -> VMobject:
    values = hough_curve_points(point, stop)
    positions = np.array([parameter_position(theta, rho) for theta, rho in values])
    return VMobject(color=color, stroke_width=2.7, stroke_opacity=opacity).set_points_as_corners(positions)


def create_parameter_marker(point, theta, color=DATA) -> Dot:
    return Dot(parameter_position(theta, compute_rho(point, theta)),
               radius=0.075, color=color).set_z_index(6)


def create_normal_geometry(point, theta) -> VGroup:
    result = VGroup(
        Line(color=ACTIVE), Arrow(buff=0, color=MUTED, tip_length=0.1),
        Arc(radius=0.43), VMobject(),
        label("ρ", [0, 0, 0], ACTIVE, 27),
        label("θ", [0, 0, 0], ACTIVE, 25),
    ).set_z_index(3)
    update_normal_geometry(result, point, theta)
    return result


def update_normal_geometry(group: VGroup, point, theta: float) -> None:
    normal = np.array([np.cos(theta), np.sin(theta), 0.0])
    tangent = np.array([-np.sin(theta), np.cos(theta), 0.0])
    rho = float(compute_rho(point, theta))
    foot = IMAGE_ORIGIN + IMAGE_UNIT * rho * normal
    visible = abs(rho) > 0.035
    # At rho=0 the normal segment vanishes; never construct a zero-length Arrow.
    safe_foot = foot if visible else IMAGE_ORIGIN + normal * 1e-5
    group[0].become(Line(IMAGE_ORIGIN, safe_foot, color=ACTIVE, stroke_width=3.5))
    group[0].set_opacity(float(visible))
    group[1].become(Arrow(IMAGE_ORIGIN, IMAGE_ORIGIN + normal * 0.78,
                          buff=0, color=MUTED, stroke_width=2, tip_length=0.11))
    group[2].become(Arc(radius=0.43, start_angle=0,
                        angle=max(theta, 1e-6), arc_center=IMAGE_ORIGIN,
                        color=ACTIVE, stroke_width=2))
    group[2].set_fill(opacity=0).set_stroke(opacity=float(theta > 0.02))
    sign = 1 if rho >= 0 else -1
    corner = [foot - tangent * 0.12,
              foot - tangent * 0.12 - sign * normal * 0.12,
              foot - sign * normal * 0.12]
    group[3].set_points_as_corners(corner).set_stroke(MUTED, 1.5, opacity=float(visible))
    group[4].move_to((IMAGE_ORIGIN + foot) / 2 + tangent * 0.27).set_opacity(float(visible))
    bisector = np.array([np.cos(theta / 2), np.sin(theta / 2), 0])
    group[5].move_to(IMAGE_ORIGIN + bisector * 0.67)


class HoughTransformPointToSinusoid(Scene):
    """A 30–38 second point → curve → shared-line explanation."""

    def play(self, *animations, **kwargs):
        super().play(*animations, **kwargs)
        # Check text after every transition/hold, including nested annotations.
        for root in self.mobjects:
            for mob in root.get_family():
                if isinstance(mob, Text) and mob.width and mob.height:
                    assert mob.get_left()[0] > -7.9 and mob.get_right()[0] < 7.9, mob.text
                    assert mob.get_bottom()[1] > -4.4 and mob.get_top()[1] < 4.4, mob.text

    def morph_text(self, attribute: str, target: Text):
        """Match existing glyphs instead of distorting whole sentences."""
        source = getattr(self, attribute)
        setattr(self, attribute, target)
        return TransformMatchingShapes(source, target)

    def caption_animation(self, text, color=WHITE, size=28):
        return self.morph_text("caption", label(text, [0, -3.82, 0], color, size))

    def set_caption(self, text, color=WHITE, duration=0.45):
        self.play(self.caption_animation(text, color), run_time=duration)

    def construct(self):
        self.rho_star, self.theta_star = common_line_parameters()
        self.theta = ValueTracker(np.deg2rad(20))
        self.caption = label("One point. How many possible lines?", [0, -3.82, 0], WHITE, 28)
        self.p1 = Dot(image_position(P1), radius=0.085, color=DATA).set_z_index(6)
        self.p1_label = label("p1 = (1, 2)", self.p1.get_center() + [-0.3, 0.65, 0], DATA, 23).set_z_index(7)
        self.active_ring = Circle(radius=0.15, color=ACTIVE, stroke_width=2).move_to(self.p1).set_z_index(5)
        self.question = label("How many lines\nthrough one point?", [3.7, 0.5, 0], WHITE, 31)
        self.candidate = create_candidate_line(P1, self.theta.get_value())
        self.add(label("Hough Transform", [0, 3.87, 0], WHITE, 42, True))
        self.play(Create(create_image_plane()),
                  FadeIn(label("Image Space", [-3.9, 2.88, 0], WHITE, 28, True)),
                  FadeIn(self.p1), FadeIn(self.p1_label), FadeIn(self.active_ring),
                  FadeIn(self.caption), FadeIn(self.question), run_time=0.8)
        self.play(Create(self.candidate), run_time=0.4)
        self.candidate.add_updater(lambda mob: mob.become(
            create_candidate_line(P1, self.theta.get_value())))
        # Three representative orientations pivot around the same fixed point.
        for degrees in (80, 135, 35):
            self.play(self.theta.animate.set_value(np.deg2rad(degrees)), run_time=0.65)
        self.set_caption("One point → infinitely many lines", DATA, duration=0.3)
        self.wait(0.55)
        self.introduce_normal_form()
        self.trace_first_point()
        self.trace_second_point()
        self.reveal_intersection()
        self.generalize()
        assert all(not mob.updaters for root in self.mobjects for mob in root.get_family())

    def introduce_normal_form(self):
        self.normal = create_normal_geometry(P1, self.theta.get_value())
        self.normal.add_updater(lambda mob: update_normal_geometry(mob, P1, self.theta.get_value()))
        self.play(FadeIn(self.normal),
                  self.morph_text("question", label("θ turns the normal\nρ is signed distance", [3.7, 1.05, 0], WHITE, 27)),
                  self.caption_animation("Measure the line by its normal", ACTIVE),
                  run_time=0.65)
        self.wait(0.6)
        # Geometry is established before the equation is introduced.
        self.equation = label("ρ = x cos(θ) + y sin(θ)", [3.7, -0.4, 0], WHITE, 27)
        self.play(FadeIn(self.equation), run_time=0.4)
        self.play(self.theta.animate.set_value(np.deg2rad(100)), run_time=1.3)
        self.wait(1.0)

    def create_readout(self, point):
        theta_text = label("θ =", [2.1, -3.1, 0], MUTED, 23)
        rho_text = label("ρ =", [4.7, -3.1, 0], MUTED, 23)
        degrees = DecimalNumber(0, num_decimal_places=1, mob_class=Text,
                                font_size=23, color=ACTIVE)
        rho = DecimalNumber(0, num_decimal_places=2, mob_class=Text,
                            font_size=23, color=ACTIVE, include_sign=True)
        unit = label("°", [3.42, -3.05, 0], MUTED, 23)
        readout = VGroup(theta_text, degrees, unit, rho_text, rho)

        def update(_mob):
            angle = self.theta.get_value()
            degrees.set_value(np.rad2deg(angle)).move_to([2.85, -3.1, 0])
            rho.set_value(compute_rho(point, angle)).move_to([5.45, -3.1, 0])

        readout.add_updater(update)
        update(readout)
        return readout

    def trace_first_point(self):
        self.play(
            Create(create_hough_plane()),
            self.morph_text("question", label("Hough Space", [3.9, 2.88, 0], WHITE, 28, True)),
            self.equation.animate.move_to([-3.9, -3.1, 0]),
            self.theta.animate.set_value(0),
            self.caption_animation("Turn the line → trace its parameters"),
            run_time=0.9,
        )
        self.readout = self.create_readout(P1)
        self.curve1 = create_curve(P1, 0)
        self.marker1 = create_parameter_marker(P1, 0)
        self.curve1.add_updater(lambda mob: mob.become(create_curve(P1, self.theta.get_value())))
        self.marker1.add_updater(lambda mob: mob.move_to(parameter_position(
            self.theta.get_value(), compute_rho(P1, self.theta.get_value()))))
        self.add(self.curve1, self.marker1, self.readout)
        self.play(self.theta.animate.set_value(THETA_END), run_time=4.5, rate_func=linear)
        self.curve1.clear_updaters()
        self.marker1.clear_updaters()
        self.readout.clear_updaters()
        self.add(label("p1", parameter_position(THETA_END, -1) + [0.35, 0, 0], DATA, 23))
        self.set_caption("One image point → one Hough curve", DATA)
        self.wait(1.4)

    def trace_second_point(self):
        self.candidate.clear_updaters()
        self.normal.clear_updaters()
        self.p2 = Dot(image_position(P2), radius=0.085, color=SECOND).set_z_index(6)
        self.p2_label = label("p2 = (3, 1)", self.p2.get_center() + [0.8, 0.30, 0], SECOND, 23).set_z_index(7)
        self.play(
            FadeIn(self.p2), FadeIn(self.p2_label),
            self.active_ring.animate.move_to(self.p2),
            Transform(self.candidate, create_candidate_line(P2, THETA_END, SECOND)),
            Transform(self.normal, create_normal_geometry(P2, THETA_END)),
            self.marker1.animate.scale(0.5).set_opacity(0.5),
            self.caption_animation("Another point → another curve", SECOND),
            run_time=0.7,
        )
        self.candidate.add_updater(lambda mob: mob.become(
            create_candidate_line(P2, self.theta.get_value(), SECOND)))
        self.normal.add_updater(lambda mob: update_normal_geometry(mob, P2, self.theta.get_value()))
        self.remove(self.readout)
        self.readout = self.create_readout(P2)
        self.add(self.readout)
        # A brief pivot on p2 precedes the second, quicker numerical sweep.
        self.play(self.theta.animate.set_value(0), run_time=0.7)
        self.curve2 = create_curve(P2, 0, SECOND)
        self.marker2 = create_parameter_marker(P2, 0, SECOND)
        self.curve2.add_updater(lambda mob: mob.become(create_curve(P2, self.theta.get_value(), SECOND)))
        self.marker2.add_updater(lambda mob: mob.move_to(parameter_position(
            self.theta.get_value(), compute_rho(P2, self.theta.get_value()))))
        self.add(self.curve2, self.marker2, self.readout)
        self.play(self.theta.animate.set_value(THETA_END), run_time=2.8, rate_func=linear)
        self.curve2.clear_updaters()
        self.marker2.clear_updaters()
        self.readout.clear_updaters()
        self.add(label("p2", parameter_position(THETA_END, -3) + [0.35, 0, 0], SECOND, 23))
        self.wait(0.55)

    def reveal_intersection(self):
        self.peak_position = parameter_position(self.theta_star, self.rho_star)
        self.remove(self.readout)
        self.readout = self.create_readout(P2)
        self.add(self.readout)
        # Markers follow their own existing curves back to the shared angle.
        self.marker1.add_updater(lambda mob: mob.move_to(parameter_position(
            self.theta.get_value(), compute_rho(P1, self.theta.get_value()))))
        self.marker2.add_updater(lambda mob: mob.move_to(parameter_position(
            self.theta.get_value(), compute_rho(P2, self.theta.get_value()))))
        self.play(self.marker1.animate.set_opacity(1).scale(2),
                  self.theta.animate.set_value(self.theta_star), run_time=1.2)
        self.candidate.clear_updaters()
        self.normal.clear_updaters()
        self.marker1.clear_updaters()
        self.marker2.clear_updaters()
        self.readout.clear_updaters()
        self.peak = Dot(self.peak_position, radius=0.09, color=RESULT).set_z_index(8)
        self.peak_ring = Circle(radius=0.22, color=RESULT, stroke_width=2).move_to(self.peak_position)
        peak_label = label("(ρ*, θ*)", self.peak_position + [0.9, 0.53, 0], RESULT, 25, True)
        peak_values = label("ρ* = 2.236   θ* = 63.435°", [3.9, -3.1, 0], RESULT, 23)
        line_label = label("L", image_position((4.65, 0.175)) + [0, 0.3, 0], RESULT, 27, True)
        self.play(
            self.candidate.animate.set_color(RESULT).set_stroke(width=4),
            self.normal.animate.set_opacity(0),
            self.active_ring.animate.set_opacity(0),
            FadeIn(self.peak), Create(self.peak_ring), FadeIn(peak_label), FadeIn(line_label),
            self.morph_text("readout", peak_values),
            self.caption_animation("Curves intersect ↔ same line", RESULT, 29),
            run_time=0.65,
        )
        self.play(Indicate(self.candidate, color=RESULT, scale_factor=1.015),
                  Indicate(self.peak_ring, color=RESULT, scale_factor=1.45), run_time=0.65)
        self.wait(2.2)

    def generalize(self):
        new_points = VGroup(*[Dot(image_position(point), radius=0.073, color=DATA).set_z_index(6)
                             for point in EXTRA_POINTS])
        self.play(LaggedStart(*[FadeIn(point, scale=0.4) for point in new_points],
                             lag_ratio=0.17),
                  self.morph_text("p1_label", label("p1", image_position(P1) + [0, 0.35, 0], DATA, 23).set_z_index(7)),
                  self.morph_text("p2_label", label("p2", image_position(P2) + [0, 0.35, 0], SECOND, 23).set_z_index(7)),
                  run_time=0.8)
        self.set_caption("Many edge pixels → votes near one (ρ, θ)", WHITE)
        # Extra curves are exact samples too; only their display opacity differs.
        curves = [create_curve(point, color=DATA, opacity=0.35) for point in EXTRA_POINTS]
        self.play(LaggedStart(*[Create(curve, rate_func=linear) for curve in curves],
                             lag_ratio=0.18), run_time=2.4)
        self.play(self.peak_ring.animate.set_stroke(width=4),
                  Indicate(self.peak, color=RESULT, scale_factor=1.6), run_time=0.7)
        self.wait(2.7)
