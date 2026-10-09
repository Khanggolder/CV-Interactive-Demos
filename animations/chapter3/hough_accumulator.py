"""Continue the approved Hough scene: actual sampled votes form an accumulator.

Each point casts one vote at each theta-bin center; rho is assigned to its
containing half-open bin, equivalently the nearest rho-bin center (ties upward).
The final line comes from the winning bin center, not an unquantized fit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import (
    Circle, Create, Dot, FadeIn, Indicate, Integer, LaggedStart, Rectangle,
    Scene, Text, Transform, TransformMatchingShapes, UpdateFromAlphaFunc,
    VGroup, config, linear,
)

# Reuse the approved scene's read-only coordinate/style helpers. No prior scene
# is instantiated; override its text-cache destination before creating text.
from hough_transform import (
    ACTIVE, BACKGROUND, DATA, EXTRA_POINTS, GRID, HOUGH_WIDTH, MUTED, P1, P2,
    RESULT, RHO_UNIT, THETA_END, WHITE, common_line_parameters, compute_rho,
    create_candidate_line, create_curve, create_hough_plane, create_image_plane,
    image_position, label, parameter_position,
)


ROOT = Path(__file__).resolve().parent
config.background_color = BACKGROUND
config.frame_width = 16
config.frame_height = 9
config.media_dir = str(ROOT / "media")
config.text_dir = str(ROOT / "media/hough_accumulator/texts")

MAIN_POINTS = np.vstack((P1, P2, EXTRA_POINTS))
DISTRACTORS = np.array([[0.5, 0.3], [2.5, 3.3], [4.1, 2.6]])
THETA_EDGES = np.deg2rad(np.arange(0, 181, 10, dtype=float))
THETA_SAMPLES = (THETA_EDGES[:-1] + THETA_EDGES[1:]) / 2
RHO_EDGES = np.arange(-5.0, 5.01, 0.5)
RHO_CENTERS = (RHO_EDGES[:-1] + RHO_EDGES[1:]) / 2
SHAPE = (len(RHO_CENTERS), len(THETA_SAMPLES))
CELL_WIDTH = HOUGH_WIDTH / SHAPE[1]
CELL_HEIGHT = RHO_UNIT * 0.5
EXAMPLE_CELLS = ((12, 0), (14, 6), (8, 17))
# One fixed scale for the entire animation. Hue changes only at higher counts.
VOTE_COLORS = ("#111A2C", "#244258", "#346D89", "#47A2C0",
               "#8CC6CB", "#D8D08E", ACTIVE)


def point_votes(point, theta_samples=THETA_SAMPLES,
                rho_bins=RHO_EDGES, theta_bins=THETA_EDGES) -> np.ndarray:
    """Return one actual (rho row, theta column) vote per sampled angle."""
    rho = compute_rho(point, theta_samples)
    rows = np.searchsorted(rho_bins, rho, side="right") - 1
    columns = np.searchsorted(theta_bins, theta_samples, side="right") - 1
    if not np.isfinite(rho).all():
        raise ValueError("Non-finite rho")
    if np.any((rows < 0) | (rows >= len(rho_bins) - 1)):
        raise ValueError("Rho outside displayed accumulator; do not clip votes")
    if np.any((columns < 0) | (columns >= len(theta_bins) - 1)):
        raise ValueError("Theta outside displayed accumulator")
    return np.column_stack((rows, columns))


def accumulate(points) -> np.ndarray:
    result = np.zeros(SHAPE, dtype=int)
    for point in points:
        hits = point_votes(point)
        np.add.at(result, (hits[:, 0], hits[:, 1]), 1)
    return result


def cell_position(row: int, column: int) -> np.ndarray:
    return parameter_position(THETA_SAMPLES[column], RHO_CENTERS[row])


def create_heatmap_cell(row, column, count=0) -> Rectangle:
    return Rectangle(width=CELL_WIDTH, height=CELL_HEIGHT,
                     stroke_color=GRID, stroke_width=0.6,
                     fill_color=VOTE_COLORS[min(int(count), 6)],
                     fill_opacity=1).move_to(cell_position(row, column))


def create_accumulator_grid() -> VGroup:
    return VGroup(*[create_heatmap_cell(r, c)
                    for r in range(SHAPE[0]) for c in range(SHAPE[1])])


def increment_cell_visual(cell: Rectangle, count: int) -> None:
    cell.set_fill(VOTE_COLORS[min(int(count), 6)], opacity=1)


def map_peak_to_line(row: int, column: int):
    """Use the displayed winning bin's center, including quantization error."""
    rho, theta = float(RHO_CENTERS[row]), float(THETA_SAMPLES[column])
    foot = rho * np.array([np.cos(theta), np.sin(theta)])
    return create_candidate_line(foot, theta, RESULT), rho, theta


class HoughAccumulatorVoting(Scene):
    """Persistent image/parameter spaces; counts are computed, never scripted."""

    def play(self, *animations, **kwargs):
        super().play(*animations, **kwargs)
        for root in self.mobjects:
            for mob in root.get_family():
                if isinstance(mob, Text) and mob.width and mob.height:
                    assert mob.get_left()[0] > -7.9 and mob.get_right()[0] < 7.9, mob.text
                    assert mob.get_bottom()[1] > -4.4 and mob.get_top()[1] < 4.4, mob.text

    def morph_text(self, attribute, target):
        old = getattr(self, attribute)
        setattr(self, attribute, target)
        return TransformMatchingShapes(old, target)

    def caption_animation(self, text, color=WHITE):
        return self.morph_text("caption", label(text, [0, -3.85, 0], color, 28))

    def construct(self):
        self.accumulator = np.zeros(SHAPE, dtype=int)
        self.visual_counts = np.zeros(SHAPE, dtype=int)
        self.completed_points = []
        self.traces = VGroup()
        self.recap()
        self.discretize()
        self.single_pixel()
        self.aligned_pixels()
        self.noise_pixels()
        self.detect_peak()
        assert np.array_equal(self.accumulator, accumulate(np.vstack((MAIN_POINTS, DISTRACTORS))))
        assert all(not mob.updaters for root in self.mobjects for mob in root.get_family())

    def recap(self):
        self.add(label("Hough Accumulator", [0, 3.87, 0], WHITE, 42, True))
        self.image_plane = create_image_plane()
        self.parameter_plane = create_hough_plane()
        self.right_heading = label("Hough Space", [3.9, 2.88, 0], WHITE, 28, True)
        self.caption = label("Same line → shared parameters", [0, -3.85, 0], WHITE, 28)
        rho, theta = common_line_parameters()
        self.line = create_candidate_line(P1, theta, RESULT)
        self.dots = [Dot(image_position(point), radius=0.085, color=DATA).set_z_index(6)
                     for point in MAIN_POINTS]
        self.recap_curves = VGroup(create_curve(P1, color=DATA, opacity=0.65),
                                  create_curve(P2, color=DATA, opacity=0.65)).set_z_index(2)
        self.shared = Circle(radius=0.18, color=RESULT, stroke_width=2).move_to(
            parameter_position(theta, rho)).set_z_index(6)
        self.play(FadeIn(self.image_plane), FadeIn(self.parameter_plane),
                  FadeIn(label("Image Space", [-3.9, 2.88, 0], WHITE, 28, True)),
                  FadeIn(self.right_heading), FadeIn(self.line),
                  FadeIn(self.dots[0]), FadeIn(self.dots[1]),
                  Create(self.recap_curves), FadeIn(self.shared), FadeIn(self.caption),
                  run_time=1.1)
        self.wait(0.9)

    def discretize(self):
        self.cells = create_accumulator_grid()
        # The previous continuous intersection falls in this cell. Its final
        # status as the unique maximum is established from the votes later.
        true_rho, true_theta = common_line_parameters()
        self.shared_index = (int(np.searchsorted(RHO_EDGES, true_rho) - 1),
                             int(np.searchsorted(THETA_EDGES, true_theta) - 1))
        row, column = self.shared_index
        box = Rectangle(width=CELL_WIDTH, height=CELL_HEIGHT,
                        stroke_color=ACTIVE, stroke_width=2).move_to(cell_position(row, column)).set_z_index(6)
        self.grid_note = label("Δθ = 10°     Δρ = 0.5", [-3.9, -3.1, 0], MUTED, 24)
        self.matrix_label = label("A[ρ, θ]", [3.9, 2.46, 0], MUTED, 24)
        self.counter_groups = VGroup()
        self.counter_numbers = []
        self.counter_boxes = []
        for (r, c), x in zip(EXAMPLE_CELLS, (1.85, 4.05, 6.25)):
            text = label(f"A[{r},{c}]", [x - 0.28, -3.12, 0], MUTED, 21)
            square = Rectangle(width=0.48, height=0.52, color=MUTED,
                               stroke_width=1, fill_color=VOTE_COLORS[0], fill_opacity=1).move_to([x + 0.59, -3.12, 0])
            number = Integer(0, mob_class=Text, font_size=29, color=WHITE).move_to(square)
            self.counter_numbers.append(number)
            self.counter_boxes.append(square)
            self.counter_groups.add(VGroup(text, square, number))
        self.play(FadeIn(self.cells), Transform(self.shared, box),
                  self.parameter_plane[0].animate.set_opacity(0),
                  self.line.animate.set_color(MUTED).set_stroke(width=1.7),
                  self.morph_text("right_heading", label("Accumulator", [3.9, 2.88, 0], WHITE, 28, True)),
                  FadeIn(self.matrix_label), FadeIn(self.grid_note), FadeIn(self.counter_groups),
                  self.caption_animation("Continuous parameters → discrete cells"), run_time=1.0)
        self.wait(0.9)
        self.play(self.recap_curves.animate.set_stroke(opacity=0), run_time=0.3)
        self.remove(self.recap_curves)
        self.active_ring = Circle(radius=0.15, color=ACTIVE, stroke_width=2).move_to(self.dots[0]).set_z_index(7)
        self.candidate = create_candidate_line(P1, 0, ACTIVE)
        self.marker = Dot(radius=0.055, color=ACTIVE).set_z_index(8)
        self.flash = box.copy().set_stroke(ACTIVE, width=2.3).set_z_index(7)

    def show_counts(self, counts: np.ndarray):
        for r, c in np.argwhere(counts != self.visual_counts):
            increment_cell_visual(self.cells[int(r) * SHAPE[1] + int(c)], int(counts[r, c]))
        self.visual_counts = counts.copy()
        for (r, c), number, box in zip(EXAMPLE_CELLS, self.counter_numbers, self.counter_boxes):
            value = int(counts[r, c])
            number.set_value(value).move_to(box)
            box.set_fill(VOTE_COLORS[min(value, 6)], opacity=1)
            number.set_color(BACKGROUND if value >= 4 else WHITE)

    def vote_for_point(self, point, dot, duration=1.0, pause_at_shared=False, keep_trace=False):
        base = self.accumulator.copy()
        hits = point_votes(point)
        trace = create_curve(point, 0, ACTIVE).set_z_index(4)
        self.play(dot.animate.set_color(ACTIVE), self.active_ring.animate.move_to(dot),
                  run_time=0.2)
        self.candidate.become(create_candidate_line(point, 0, ACTIVE))
        self.marker.move_to(parameter_position(0, compute_rho(point, 0)))
        self.flash.set_opacity(0)
        self.add(trace, self.candidate, self.marker, self.flash)

        def sweep(start, stop, run_time):
            def update(mob, alpha):
                theta = start + alpha * (stop - start)
                # Rebuild from a fixed base and the angle prefix, so votes do
                # not depend on frame rate, frame skipping or repeated updates.
                count = int(np.searchsorted(THETA_SAMPLES, theta + 1e-10, side="right"))
                current = base.copy()
                np.add.at(current, (hits[:count, 0], hits[:count, 1]), 1)
                self.show_counts(current)
                mob.become(create_curve(point, theta, ACTIVE).set_z_index(4))
                self.candidate.become(create_candidate_line(point, theta, ACTIVE))
                self.marker.move_to(parameter_position(theta, compute_rho(point, theta)))
                if count:
                    r, c = hits[count - 1]
                    self.flash.move_to(cell_position(int(r), int(c))).set_stroke(opacity=1)
                    self.flash.set_fill(opacity=0)
            self.play(UpdateFromAlphaFunc(trace, update), run_time=run_time, rate_func=linear)

        if pause_at_shared:
            middle = float(THETA_SAMPLES[self.shared_index[1]])
            sweep(0, middle, duration * 0.45)
            # Both the highlighted bin and its enlarged integer counter update
            # before this pause: the first pixel gives 1, the second gives 2.
            self.wait(0.65)
            sweep(middle, THETA_END, duration * 0.55)
        else:
            sweep(0, THETA_END, duration)
        np.add.at(base, (hits[:, 0], hits[:, 1]), 1)
        self.accumulator = base
        self.show_counts(base)
        self.completed_points.append(np.asarray(point))
        assert np.array_equal(base, accumulate(self.completed_points))
        dot.set_color(DATA)
        self.remove(self.marker, self.flash, self.candidate)
        if keep_trace:
            trace.set_stroke(DATA, opacity=0.16)
            self.traces.add(trace)
        else:
            self.remove(trace)

    def single_pixel(self):
        self.play(FadeIn(self.active_ring),
                  self.caption_animation("One pixel votes for many cells", DATA),
                  self.morph_text("grid_note", label("18 sampled angles per pixel", [-3.9, -3.1, 0], MUTED, 24)),
                  run_time=0.5)
        self.vote_for_point(P1, self.dots[0], duration=3.0, pause_at_shared=True, keep_trace=True)
        self.wait(0.8)

    def aligned_pixels(self):
        self.play(self.caption_animation("Aligned pixels reinforce the shared cell"), run_time=0.45)
        self.vote_for_point(P2, self.dots[1], duration=2.2, pause_at_shared=True, keep_trace=True)
        self.wait(0.5)
        for point, dot in zip(MAIN_POINTS[2:], self.dots[2:]):
            self.play(FadeIn(dot, scale=0.4), run_time=0.2)
            self.vote_for_point(point, dot, duration=0.85)
        self.play(Indicate(self.counter_boxes[1], color=ACTIVE, scale_factor=1.15), run_time=0.5)
        self.wait(0.8)

    def noise_pixels(self):
        noise_dots = [Dot(image_position(point), radius=0.075, color=DATA).set_z_index(6)
                      for point in DISTRACTORS]
        self.play(LaggedStart(*[FadeIn(dot, scale=0.4) for dot in noise_dots], lag_ratio=0.2),
                  self.caption_animation("Unaligned pixels spread their votes"), run_time=0.6)
        for point, dot in zip(DISTRACTORS, noise_dots):
            self.vote_for_point(point, dot, duration=0.75)
        self.wait(0.8)

    def detect_peak(self):
        winners = np.argwhere(self.accumulator == self.accumulator.max())
        assert len(winners) == 1, "The teaching example requires one unique maximum"
        row, column = map(int, winners[0])
        assert (row, column) == self.shared_index
        detected, rho, theta = map_peak_to_line(row, column)
        assert int(self.accumulator[row, column]) == 6
        peak_cell = self.cells[row * SHAPE[1] + column]
        self.play(self.shared.animate.set_stroke(RESULT, width=3),
                  peak_cell.animate.set_fill(RESULT),
                  self.active_ring.animate.set_opacity(0),
                  self.traces.animate.set_stroke(opacity=0),
                  self.counter_boxes[1].animate.set_fill(RESULT).set_stroke(RESULT, width=2),
                  self.caption_animation("Peak = strongest line support", RESULT),
                  run_time=0.6)
        peak_label = label("Peak: 6 votes", cell_position(row, column) + [1.25, 0.52, 0], RESULT, 25, True)
        self.play(FadeIn(peak_label),
                  Indicate(self.shared, color=RESULT, scale_factor=1.35), run_time=0.6)
        self.wait(0.7)
        self.play(Transform(self.line, detected.set_stroke(width=4)),
                  self.caption_animation("Accumulator peak ↔ detected line", RESULT),
                  self.morph_text("grid_note", label(
                      f"Bin center: ρ = {rho:.2f}, θ = {np.rad2deg(theta):.0f}°",
                      [-3.9, -3.1, 0], RESULT, 24)), run_time=0.9)
        self.play(self.morph_text("matrix_label", label("Bin-center approximation", [3.9, 2.46, 0], MUTED, 23)),
                  run_time=0.4)
        self.wait(0.8)
        self.play(self.caption_animation("Many aligned pixels → one accumulator peak"), run_time=0.45)
        self.wait(2.1)
