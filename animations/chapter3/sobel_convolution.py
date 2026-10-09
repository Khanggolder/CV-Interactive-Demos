"""Sobel: the same pixels become products, a sum, and a gradient.

Chapter 3's displayed masks are applied directly (cross-correlation, no flip).
Array rows and image-coordinate y increase DOWNWARD. Display coordinates map
(Gx, Gy) to (Gx, -Gy). This example has exactly Gy=0: the component triangle
is genuinely degenerate, not an invented nonzero vertical component.
"""

from pathlib import Path

import numpy as np
from manim import (
    Arrow, Circle, Create, Dot, FadeIn, FadeOut, Indicate, Line, ManimColor,
    Rectangle, ReplacementTransform, Scene, Succession, Text, Transform,
    TransformFromCopy, VGroup, config, interpolate_color,
)

ROOT = Path(__file__).resolve().parent
BG, WHITE, MUTED = "#0B1020", "#F8FAFC", "#8892A8"
CYAN, ORANGE, PURPLE = "#47C2FF", "#FF8C42", "#B392F0"
YELLOW, GREEN, GRID = "#FFD166", "#69D394", "#283248"
KERNEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
KERNEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
PATCH_CENTER = (2, 3)  # [row, column], zero based.
CELL = 1.02
LEFT = np.array([-3.65, 0.35, 0.0])
RIGHT = np.array([2.8, 0.35, 0.0])
VECTOR_SCALE = 3.1 / 720  # Equal scale for both components; geometry only.
config.background_color = BG
config.frame_width, config.frame_height = 16, 9
config.media_dir = str(ROOT / "media")
config.text_dir = str(ROOT / "media/sobel/texts")


def generate_sobel_example():
    image = np.full((5, 7), 20, dtype=np.int64)
    image[:, 4:] = 200
    return image


def compute_sobel(image, center=PATCH_CENTER):
    row, col = center
    patch = image[row - 1:row + 2, col - 1:col + 2].copy()
    assert patch.shape == (3, 3)
    products_x, products_y = patch * KERNEL_X, patch * KERNEL_Y
    gx, gy = int(products_x.sum()), int(products_y.sum())
    return dict(patch=patch, products_x=products_x, products_y=products_y,
                gx=gx, gy=gy, magnitude=float(np.hypot(gx, gy)),
                theta_radians=float(np.arctan2(gy, gx)),
                theta_degrees=float(np.degrees(np.arctan2(gy, gx))))


def label(text, at, color=WHITE, size=29):
    point = np.array([*at, 0.0]) if len(at) == 2 else at
    return Text(text, font_size=size, color=color).move_to(point)


def cell_position(row, col, center=LEFT, cell=CELL):
    return center + cell * np.array([col - 1, 1 - row, 0.0])


def create_pixel_grid(image, center=LEFT, cell=0.65):
    h, w = image.shape
    return VGroup(*[
        Rectangle(width=cell, height=cell, stroke_width=0,
                  fill_opacity=1, fill_color=interpolate_color(
                      ManimColor("#10192B"), ManimColor("#B5C4D4"), float(image[r, c] / 255)))
        .move_to(center + cell * np.array([c - (w - 1) / 2, (h - 1) / 2 - r, 0]))
        for r in range(h) for c in range(w)
    ])


def create_kernel_grid(kernel, color, center=RIGHT):
    boxes = VGroup(*[Rectangle(width=CELL, height=CELL, stroke_color=color,
                               stroke_width=1.5, fill_opacity=0)
                     .move_to(cell_position(r, c, center))
                     for r in range(3) for c in range(3)])
    numbers = VGroup(*[label(f"{int(kernel[r, c]):+d}" if kernel[r, c] else "0",
                            cell_position(r, c, center), color, 32)
                       for r in range(3) for c in range(3)])
    return boxes, numbers


def product_label(value, row, col):
    return label(str(int(value)), cell_position(row, col, RIGHT),
                 GREEN if value > 0 else ORANGE if value < 0 else MUTED, 32)


def screen_vector(gx, gy):
    return VECTOR_SCALE * np.array([gx, -gy, 0.0])


class TextSwap(Succession):
    """Remove the entire old caption before bringing in the new sentence."""

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


class SobelConvolution(Scene):
    def play(self, *animations, **kwargs):
        super().play(*animations, **kwargs)
        # Also exercised by the skip-animation preflight, before video render.
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
        new = label(text, (0, -3.35), size=30)
        assert new.get_top()[1] < -3.0 and new.get_bottom()[1] > -3.7
        return self.swap("caption", new)

    def progress(self, index):
        return self.marker.animate.move_to([self.steps[index].get_x(), -4.24, 0])

    def construct(self):
        self.image = generate_sobel_example()
        self.data = compute_sobel(self.image)
        self.title = label("Sobel · measure a local change", (0, 3.8), size=36)
        self.caption = label("Start with an intensity boundary", (0, -3.35), size=30)
        self.steps = VGroup(*[label(name, (x, -3.97), MUTED, 22)
                             for x, name in zip(np.linspace(-6, 6, 6),
                                                ["Pixels", "Gx", "Gy", "Vector", "M / θ", "Slide"])])
        self.marker = Dot([-6, -4.24, 0], radius=0.045, color=CYAN)
        self.add(self.title, self.caption, self.steps, self.marker)
        self.image_to_pixels()
        self.apply_gx()
        self.explain_gx()
        self.apply_gy()
        self.build_vector()
        self.magnitude_direction()
        self.slide_and_finish()

    def image_to_pixels(self):
        self.grid = create_pixel_grid(self.image)
        self.play(FadeIn(self.grid), run_time=0.6)
        self.wait(0.45)
        self.active = Rectangle(width=1.95, height=1.95, color=CYAN, stroke_width=3).move_to(LEFT)
        self.play(Create(self.active), run_time=0.4)
        target = create_pixel_grid(self.image, cell=CELL)
        for r in range(5):
            for c in range(7):
                target[r * 7 + c].set_stroke(GRID, 1)
                if not (1 <= r <= 3 and 2 <= c <= 4):
                    target[r * 7 + c].set_opacity(0.23)
        self.play(Transform(self.grid, target),
                  self.active.animate.set(width=3 * CELL, height=3 * CELL),
                  self.cue("Pixels are numbers"), run_time=0.8)
        self.pixel_numbers = VGroup(*[
            label(str(int(self.data["patch"][r, c])), cell_position(r, c),
                  BG if self.data["patch"][r, c] > 128 else WHITE, 32)
            for r in range(3) for c in range(3)])
        self.patch_title = label("Same 3 × 3 patch", (-3.65, 2.6), CYAN, 28)
        self.play(FadeIn(self.pixel_numbers), FadeIn(self.patch_title), run_time=0.45)
        self.wait(0.5)

    def apply_gx(self):
        self.kernel_boxes, self.weights = create_kernel_grid(KERNEL_X, ORANGE)
        self.weights.set_z_index(5)
        self.panel_title = label("Gx mask", (2.8, 2.6), ORANGE, 30)
        self.play(Create(self.kernel_boxes), FadeIn(self.weights), FadeIn(self.panel_title),
                  self.progress(1), run_time=0.6)
        # Values and weights occupy different subregions in each persistent cell.
        self.weight_backplates = VGroup(*[
            Rectangle(width=0.55, height=0.40, stroke_width=0,
                      fill_color=BG, fill_opacity=0.97)
            .move_to(cell_position(i // 3, i % 3) + [0.22, 0.23, 0])
            .set_z_index(4) for i in range(9)])
        self.play(self.kernel_boxes.animate.move_to(LEFT),
                  FadeIn(self.weight_backplates),
                  *[self.weights[i].animate.scale(0.78).move_to(
                      cell_position(i // 3, i % 3) + [0.22, 0.23, 0]) for i in range(9)],
                  *[self.pixel_numbers[i].animate.shift([-0.13, -0.21, 0]) for i in range(9)],
                  self.cue("Pair each pixel with its weight"), run_time=0.65)
        self.play(self.swap("panel_title", label("Nine products", (2.8, 2.6), ORANGE, 30)), run_time=0.45)
        self.products = VGroup()
        self.expression = label("value × weight", (2.8, -1.75), YELLOW, 30)
        self.add(self.expression)
        for i in range(9):
            r, c = divmod(i, 3)
            value, weight = int(self.data["patch"][r, c]), int(KERNEL_X[r, c])
            result = product_label(self.data["products_x"][r, c], r, c)
            if i < 3:
                expression = label(f"{value} × ({weight:+d})", (2.8, -1.75), YELLOW, 32)
                self.play(self.swap("expression", expression),
                          Indicate(self.pixel_numbers[i], color=YELLOW),
                          Indicate(self.weights[i], color=YELLOW), run_time=0.45)
                self.play(TransformFromCopy(expression, result), run_time=0.3)
                self.wait(0.15)
            else:
                pair = VGroup(self.pixel_numbers[i], self.weights[i])
                self.play(TransformFromCopy(pair, result), run_time=0.15)
            self.products.add(result)
        self.play(FadeOut(self.expression), self.cue("Multiply all nine pairs, then add"), run_time=0.45)
        self.running = label("Σ = 0", (2.8, -2.3), ORANGE, 32)
        self.add(self.running)
        total = 0
        for r in range(3):
            total += int(self.data["products_x"][r].sum())
            # Preserve all nine products; copies flow into the cumulative sum.
            dest = label(f"Σ = {total}", (2.8, -2.3), ORANGE, 32)
            self.play(TransformFromCopy(VGroup(*self.products[3*r:3*r+3]), dest),
                      FadeOut(self.running), run_time=0.3)
            self.running = dest
        self.gx_label = label(f"Gx = {self.data['gx']}", (2.8, -2.3), ORANGE, 34)
        self.play(TextSwap(self.running, self.gx_label), run_time=0.45)
        self.wait(0.6)

    def explain_gx(self):
        edge_x = LEFT[0] + CELL / 2
        self.edge = Line([edge_x, -2.05, 0], [edge_x, 2.25, 0], color=CYAN, stroke_width=2)
        self.crossing = Arrow([edge_x - 0.8, -1.75, 0], [edge_x + 1.15, -1.75, 0],
                              buff=0, color=ORANGE, stroke_width=4)
        self.play(Create(self.edge), Create(self.crossing),
                  self.cue("Large Gx: strong left-to-right change"), run_time=0.6)
        self.wait(0.9)

    def apply_gy(self):
        _, new_weights = create_kernel_grid(KERNEL_Y, PURPLE)
        new_weights.set_z_index(5)
        for i, mob in enumerate(new_weights):
            mob.scale(0.78).move_to(cell_position(i // 3, i % 3) + [0.22, 0.23, 0])
        self.play(Transform(self.weights, new_weights), self.kernel_boxes.animate.set_color(PURPLE),
                  self.swap("panel_title", label("Gy · same nine pixels", (2.8, 2.6), PURPLE, 29)),
                  FadeOut(self.crossing), self.progress(2),
                  self.cue("Now compare top and bottom"), run_time=0.6)
        for r in range(3):
            targets = [product_label(self.data["products_y"][r, c], r, c) for c in range(3)]
            self.play(*[TextSwap(self.products[r * 3 + c], targets[c]) for c in range(3)],
                      *[Indicate(self.weights[r * 3 + c], color=YELLOW) for c in range(3)],
                      run_time=0.45)
            for c, target in enumerate(targets):
                self.products[r * 3 + c] = target
        self.gy_label = label(f"Gy = {self.data['gy']}", (4.7, -2.3), PURPLE, 34)
        self.play(self.gx_label.animate.move_to([1.25, -2.3, 0]),
                  TransformFromCopy(self.products, self.gy_label),
                  self.cue("Equal rows: no vertical intensity change"), run_time=0.5)
        self.wait(0.75)

    def build_vector(self):
        # The same image shrinks; no second unrelated image is substituted.
        small = create_pixel_grid(self.image, center=np.array([-4.3, 0.45, 0]), cell=0.65)
        self.small_edge_x = -4.3 + 0.65 / 2
        self.play(Transform(self.grid, small), FadeOut(self.weights), FadeOut(self.kernel_boxes),
                  FadeOut(self.weight_backplates),
                  FadeOut(self.pixel_numbers), FadeOut(self.products), FadeOut(self.active),
                  FadeOut(self.panel_title), FadeOut(self.edge),
                  self.swap("patch_title", label("Image boundary", (-4.3, 2.6), CYAN, 28)),
                  self.gx_label.animate.move_to([1.4, 2.6, 0]),
                  self.gy_label.animate.move_to([4.8, 2.6, 0]),
                  self.cue("Two responses form one gradient vector"), self.progress(3), run_time=0.8)
        self.origin = np.array([1.05, 0.55, 0.0])
        self.endpoint = self.origin + screen_vector(self.data["gx"], self.data["gy"])
        self.axes = VGroup(
            Arrow(self.origin + [-0.3, 0, 0], self.origin + [4.2, 0, 0], buff=0, color=MUTED, stroke_width=1.5),
            Arrow(self.origin + [0, 1.0, 0], self.origin + [0, -1.45, 0], buff=0, color=MUTED, stroke_width=1.5),
            label("x", (5.55, 0.55), MUTED, 24), label("y ↓", (1.05, -1.25), MUTED, 24))
        self.vector = Arrow(self.origin, self.endpoint, buff=0, color=YELLOW, stroke_width=6)
        self.vector_text = label(f"∇I = ({self.data['gx']}, {self.data['gy']})", (3.1, 1.65), YELLOW, 31)
        self.play(Create(self.axes), TransformFromCopy(VGroup(self.gx_label, self.gy_label), self.vector_text),
                  Create(self.vector), run_time=0.65)
        self.edge = Line([self.small_edge_x, -1.2, 0], [self.small_edge_x, 2.1, 0], color=CYAN, stroke_width=2)
        image_origin = np.array([self.small_edge_x - 0.7, 0.45, 0])
        image_end = image_origin + screen_vector(self.data["gx"], self.data["gy"]) * 0.62
        self.image_arrow = Arrow(image_origin, image_end, buff=0, color=YELLOW, stroke_width=5)
        corner = np.array([self.small_edge_x, 0.45, 0])
        self.right_angle = VGroup(Line(corner + [0, 0.22, 0], corner + [0.22, 0.22, 0], color=WHITE),
                                  Line(corner + [0.22, 0.22, 0], corner + [0.22, 0, 0], color=WHITE))
        self.play(Create(self.edge), TransformFromCopy(self.vector, self.image_arrow),
                  Create(self.right_angle), run_time=0.55)
        self.wait(0.55)

    def magnitude_direction(self):
        # Gy=0 means endpoint == the right-angle corner: a collapsed triangle.
        self.component = Line(self.origin, self.endpoint, color=ORANGE, stroke_width=9).set_z_index(-1)
        self.zero = Circle(radius=0.085, color=PURPLE, stroke_width=2).move_to(self.endpoint)
        self.triangle_note = label("Gy = 0: triangle lies flat", (3.6, -0.85), PURPLE, 26)
        self.play(TransformFromCopy(self.gx_label, self.component), Create(self.zero),
                  FadeIn(self.triangle_note), self.progress(4),
                  self.cue("Magnitude is the vector's length"), run_time=0.6)
        self.magnitude = label(f"M = √(Gx² + Gy²) = {self.data['magnitude']:.0f}", (2.85, -1.95), GREEN, 30)
        self.play(FadeIn(self.magnitude), self.vector.animate.set_color(GREEN), run_time=0.5)
        self.wait(0.9)
        # Zero angle is shown by coincident rays and an explicit 0° label, not
        # a misleading visible nonzero angle arc.
        self.angle_label = label(f"θ = {self.data['theta_degrees']:.0f}°", (4.55, 1.0), YELLOW, 28)
        self.direction = label(f"θ = atan2(Gy, Gx) = {self.data['theta_degrees']:.0f}°", (2.85, -2.6), YELLOW, 29)
        self.play(FadeIn(self.direction), FadeIn(self.angle_label),
                  self.cue("Gradient points across the edge · 90° to it"), run_time=0.5)
        self.play(Indicate(self.right_angle, color=YELLOW),
                  self.image_arrow.animate.set_color(GREEN), run_time=0.5)
        self.wait(1.0)

    def slide_and_finish(self):
        # Reuse the same grid; center advances exactly one source column.
        self.window = Rectangle(width=1.95, height=1.95, color=CYAN, stroke_width=3)
        self.window.move_to([-4.3, 0.45, 0])
        self.center_dot = Dot([-4.3, 0.45, 0], radius=0.06, color=YELLOW)
        self.play(FadeOut(self.image_arrow), FadeOut(self.right_angle), FadeOut(self.edge),
                  Create(self.window), FadeIn(self.center_dot), self.progress(5),
                  self.cue("Slide → multiply → sum"), run_time=0.6)
        self.shift_result = label("Center pixel: column 3", (-4.3, -1.65), CYAN, 25)
        self.play(FadeIn(self.shift_result), run_time=0.3)
        self.play(self.window.animate.shift([0.65, 0, 0]),
                  self.center_dot.animate.shift([0.65, 0, 0]),
                  self.swap("shift_result", label("Center pixel: column 4", (-4.3, -1.65), CYAN, 25)), run_time=0.7)
        shifted = compute_sobel(self.image, (2, 4))
        self.shift_value = label(f"Gx = {shifted['gx']}   Gy = {shifted['gy']}", (-4.3, -2.3), WHITE, 28)
        self.play(FadeIn(self.shift_value), self.cue("Same operation at each interior pixel"), run_time=0.5)
        self.wait(0.7)
        self.play(FadeOut(self.triangle_note), FadeOut(self.angle_label), FadeOut(self.axes),
                  FadeOut(self.shift_result), FadeOut(self.shift_value), FadeOut(self.component), FadeOut(self.zero),
                  self.cue("Patch → Gx, Gy → gradient → M, θ"),
                  self.swap("patch_title", label("Sobel: local gradient", (-4.3, 2.6), CYAN, 29)), run_time=0.6)
        self.wait(1.6)
