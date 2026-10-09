"""Visual-first prototype of the Canny edge-detection pipeline."""

from __future__ import annotations

from collections import deque

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Dot,
    FadeIn,
    ImageMobject,
    LaggedStart,
    Line,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    Square,
    UpdateFromAlphaFunc,
    RESAMPLING_ALGORITHMS,
    Text,
    Transform,
    VGroup,
    WHITE,
    config,
    linear,
)


BACKGROUND = "#0B1020"
PANEL = "#121A2F"
MUTED = "#8892A8"
SMOOTHING = "#47C2FF"
GRADIENT = "#B68CFF"
STRONG = "#FFD166"
WEAK = "#FF8C42"
REJECTED = "#3D465C"
EDGE = "#F8FAFC"

config.background_color = BACKGROUND


def gaussian_kernel(size: int = 5, sigma: float = 1.15) -> np.ndarray:
    """Return a normalized square Gaussian kernel."""
    radius = size // 2
    axis = np.arange(-radius, radius + 1, dtype=float)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Small dependency-free 2-D convolution with reflected borders."""
    kh, kw = kernel.shape
    py, px = kh // 2, kw // 2
    padded = np.pad(image, ((py, py), (px, px)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel, optimize=True)


def synthetic_input(size: int = 224) -> np.ndarray:
    """Create a deterministic teaching image with texture and clear geometry."""
    yy, xx = np.mgrid[:size, :size]
    image = 24 + 18 * (xx / size) + 10 * (yy / size)

    circle = (xx - 72) ** 2 + (yy - 80) ** 2 <= 39**2
    image[circle] = 164

    rectangle = (xx > 116) & (xx < 194) & (yy > 104) & (yy < 181)
    image[rectangle] = 102

    diagonal = np.abs(0.72 * xx + yy - 226) < 4
    image[diagonal] = 218

    ring = np.abs(np.sqrt((xx - 163) ** 2 + (yy - 60) ** 2) - 24) < 3
    image[ring] = 205

    rng = np.random.default_rng(13)
    noise = rng.normal(0, 13, (size, size))
    texture = 6 * np.sin(xx / 3.7) * np.sin(yy / 5.1)
    return np.clip(image + noise + texture, 0, 255).astype(np.uint8)


def canny_stages(image: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the intermediate arrays used by the animation."""
    source = image.astype(float) / 255.0
    smoothed = convolve(source, gaussian_kernel())

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = sobel_x.T
    gx = convolve(smoothed, sobel_x)
    gy = convolve(smoothed, sobel_y)
    magnitude = np.hypot(gx, gy)
    magnitude /= max(float(np.percentile(magnitude, 99.5)), 1e-9)
    magnitude = np.clip(magnitude, 0, 1)
    direction = np.arctan2(gy, gx)

    nms = non_maximum_suppression(magnitude, direction)
    nms /= max(float(nms.max()), 1e-9)
    strong, weak, edges = hysteresis(nms, low=0.10, high=0.35)

    return {
        "original": source,
        "smoothed": smoothed,
        "magnitude": magnitude,
        "direction": direction,
        "nms": nms,
        "strong": strong,
        "weak": weak,
        "edges": edges,
    }


def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Thin ridges by retaining maxima along the local gradient direction."""
    result = np.zeros_like(magnitude)
    angle = (np.rad2deg(direction) + 180) % 180

    center = magnitude[1:-1, 1:-1]
    sectors = (
        ((angle[1:-1, 1:-1] < 22.5) | (angle[1:-1, 1:-1] >= 157.5), (0, 1)),
        (((angle[1:-1, 1:-1] >= 22.5) & (angle[1:-1, 1:-1] < 67.5)), (1, 1)),
        (((angle[1:-1, 1:-1] >= 67.5) & (angle[1:-1, 1:-1] < 112.5)), (1, 0)),
        (((angle[1:-1, 1:-1] >= 112.5) & (angle[1:-1, 1:-1] < 157.5)), (1, -1)),
    )

    for mask, (dy, dx) in sectors:
        forward = magnitude[1 + dy : magnitude.shape[0] - 1 + dy, 1 + dx : magnitude.shape[1] - 1 + dx]
        backward = magnitude[1 - dy : magnitude.shape[0] - 1 - dy, 1 - dx : magnitude.shape[1] - 1 - dx]
        keep = mask & (center >= forward) & (center >= backward)
        result[1:-1, 1:-1][keep] = center[keep]
    return result


def hysteresis(
    nms: np.ndarray, low: float, high: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep weak pixels only when they connect to a strong edge."""
    strong = nms >= high
    weak = (nms >= low) & ~strong
    connected = strong.copy()
    queue = deque(map(tuple, np.argwhere(strong)))
    height, width = nms.shape

    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and weak[ny, nx]
                    and not connected[ny, nx]
                ):
                    connected[ny, nx] = True
                    queue.append((ny, nx))
    return strong, weak, connected


def grayscale_rgb(image: np.ndarray) -> np.ndarray:
    values = np.clip(image * 255, 0, 255).astype(np.uint8)
    return np.repeat(values[..., None], 3, axis=2)


def threshold_rgb(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    output = np.zeros((*strong.shape, 3), dtype=np.uint8)
    output[:] = (16, 22, 38)
    output[weak] = (255, 140, 66)
    output[strong] = (255, 209, 102)
    return output


class CannyPipelinePrototype(Scene):
    """A visual-first tour through the full Canny pipeline."""

    stage_names = (
        "Original",
        "Smooth",
        "Gradient",
        "NMS",
        "Hysteresis",
        "Final",
    )
    stage_colors = (WHITE, SMOOTHING, GRADIENT, STRONG, WEAK, EDGE)

    def construct(self) -> None:
        arrays = canny_stages(synthetic_input())
        # The same image and magnified pixel grid persist through the first four
        # stages. Both always use arrays from the numerical pipeline above.
        title = Text("Canny Edge Detector", font_size=42, weight="BOLD", color=WHITE)
        title.to_edge(UP, buff=0.32)
        self.add(title)
        overview = self.pipeline_overview()
        self.play(LaggedStart(*[FadeIn(item) for item in overview], lag_ratio=0.05), run_time=1.0)
        self.wait(0.4)
        rail, dots = self.progress_rail()
        self.play(ReplacementTransform(overview, rail), run_time=0.7)

        image, border = self.image_panel(grayscale_rgb(arrays["original"]))
        stage_label = self.new_stage_label("Original image", WHITE, border)
        # This patch straddles the rectangle's left boundary. The NMS examples
        # below are actual adjacent pixels, not hand-picked illustrative values.
        center = (130, 116)
        crop = self.patch_slice(center)
        grid = self.pixel_grid(grayscale_rgb(arrays["original"])[crop])
        heading = self.side_title("Look closely at this boundary", WHITE)
        note = Text("one square = one image pixel", font_size=23, color=MUTED).move_to([3.55, -1.25, 0])
        formula = Text("intensity changes across the edge", font_size=23, color=WHITE).move_to([3.55, -2.35, 0])
        locator = self.patch_locator(center, WHITE)
        connector = Line(locator.get_right(), grid.get_left(), color=REJECTED, stroke_width=1.5)
        self.play(FadeIn(border), FadeIn(image), FadeIn(stage_label),
                  FadeIn(grid), FadeIn(heading), FadeIn(note), FadeIn(formula),
                  FadeIn(locator), FadeIn(connector), run_time=0.6)
        self.play(dots[0].animate.set_fill(WHITE, 1), run_time=0.2)
        self.wait(0.6)

        self.play(
            Transform(image, self.new_image(grayscale_rgb(arrays["smoothed"]))),
            Transform(grid, self.pixel_grid(grayscale_rgb(arrays["smoothed"])[crop])),
            Transform(stage_label, self.new_stage_label("1 · Gaussian smoothing", SMOOTHING, border)),
            Transform(heading, self.side_title("Neighbors smooth the noise", SMOOTHING)),
            Transform(note, Text("near pixels carry more weight", font_size=23, color=MUTED).move_to(note)),
            Transform(formula, Text("Iσ = Gσ ∗ I", font_size=28, color=SMOOTHING).move_to(formula)),
            locator.animate.set_color(SMOOTHING),
            dots[1].animate.set_fill(SMOOTHING, 1), run_time=1.2,
        )
        self.wait(1.3)

        gradient_pixels = grayscale_rgb(arrays["magnitude"])
        self.play(
            Transform(image, self.new_image(gradient_pixels)),
            Transform(grid, self.pixel_grid(gradient_pixels[crop])),
            Transform(stage_label, self.new_stage_label("2 · Magnitude + direction", GRADIENT, border)),
            Transform(heading, self.side_title("Across the boundary", GRADIENT)),
            Transform(note, Text("brightness = gradient magnitude", font_size=23, color=MUTED).move_to(note)),
            Transform(formula, Text("M = √(Gx² + Gy²)", font_size=23, color=GRADIENT).move_to(formula)),
            locator.animate.set_color(GRADIENT),
            dots[2].animate.set_fill(GRADIENT, 1), run_time=1.2,
        )
        theta = float(arrays["direction"][center])
        # Image rows increase downward; Manim y increases upward.
        vector = np.array([np.cos(theta), -np.sin(theta), 0])
        arrow = Arrow(grid.get_center() - vector * 0.9, grid.get_center() + vector * 0.9,
                      buff=0, color=GRADIENT, stroke_width=4)
        self.play(FadeIn(arrow), run_time=0.3)
        self.wait(1.3)

        # A real local maximum and its shoulder, in the same displayed patch.
        keep = center
        delete = (130, 115)
        assert arrays["nms"][keep] > 0 and arrays["nms"][delete] == 0
        comparison = self.nms_comparison(arrays, keep)
        keep_marks = self.neighbor_marks(arrays, keep, center)
        self.play(
            Transform(stage_label, self.new_stage_label("3 · Non-maximum suppression", STRONG, border)),
            Transform(heading, self.side_title("Compare q1, p, q2 along θ", STRONG)),
            Transform(note, comparison),
            Transform(formula, Text("p is largest → KEEP", font_size=26, color=STRONG).move_to(formula)),
            Transform(arrow, keep_marks),
            locator.animate.set_color(STRONG),
            dots[3].animate.set_fill(STRONG, 1), run_time=0.6,
        )
        self.wait(1.7)
        self.play(
            Transform(note, self.nms_comparison(arrays, delete)),
            Transform(arrow, self.neighbor_marks(arrays, delete, center)),
            Transform(formula, Text("a neighbor is larger → DELETE", font_size=26, color=WEAK).move_to(formula)),
            run_time=0.55,
        )
        self.wait(1.7)
        thin_pixels = grayscale_rgb(arrays["nms"])
        # Retained ridge pixels hold position while suppressed shoulders darken.
        self.play(
            Transform(image, self.new_image(thin_pixels)),
            Transform(grid, self.pixel_grid(thin_pixels[crop])),
            arrow.animate.set_opacity(0),
            Transform(note, Text("thick response → one-pixel ridge", font_size=23, color=MUTED).move_to([3.55, -1.25, 0])),
            Transform(formula, Text("M(p) ≥ M(q1) and M(p) ≥ M(q2)", font_size=23, color=STRONG).move_to(formula)),
            run_time=1.2,
        )
        self.wait(1.3)

        distances = self.propagation_distances(arrays)
        candidates = [tuple(p) for p in np.argwhere(distances == 3)
                      if np.all((p >= 4) & (p < 220))]
        connected_center = max(candidates, key=lambda p: int(
            (arrays["weak"] & arrays["edges"])[self.patch_slice(p)].sum()))
        isolated = np.argwhere(arrays["weak"] & ~arrays["edges"])
        # Choose a genuine isolated weak candidate, away from image borders.
        isolated = isolated[np.all((isolated >= 4) & (isolated < 220), axis=1)]
        isolated_center = tuple(isolated[len(isolated) // 2])
        connected_crop = self.patch_slice(connected_center)
        isolated_crop = self.patch_slice(isolated_center)
        threshold_pixels = threshold_rgb(arrays["strong"], arrays["weak"])
        connected_position = np.array([2.0, 0.35, 0])
        isolated_position = np.array([5.1, 0.35, 0])
        connected_grid = self.pixel_grid(threshold_pixels[connected_crop], 0.28, connected_position)
        isolated_grid = self.pixel_grid(threshold_pixels[isolated_crop], 0.28, isolated_position)
        second_locator = self.patch_locator(isolated_center, WEAK)
        captions = VGroup(
            Text("connected weak", font_size=22, color=WEAK).move_to([2.0, -0.95, 0]),
            Text("isolated weak", font_size=22, color=WEAK).move_to([5.1, -0.95, 0]),
        )
        legend = self.hysteresis_legend()
        self.play(
            Transform(image, self.new_image(threshold_pixels)),
            Transform(grid, connected_grid),
            FadeIn(isolated_grid), FadeIn(second_locator),
            Transform(locator, self.patch_locator(connected_center, STRONG)),
            connector.animate.set_opacity(0),
            Transform(stage_label, self.new_stage_label("4 · Hysteresis", WEAK, border)),
            Transform(heading, self.side_title("Follow 8-connected neighbors", WEAK)),
            Transform(note, captions),
            Transform(formula, legend),
            dots[4].animate.set_fill(WEAK, 1), run_time=0.8,
        )
        self.wait(0.9)

        def propagate(_mob, alpha):
            pixels = threshold_pixels.copy()
            depth = (alpha * 4 if alpha <= 0.75 else
                     3 + (alpha - 0.75) * 4 * (distances.max() - 3))
            reached = (distances >= 0) & (distances <= depth)
            # Gold remains strong. White marks accepted weak pixels without
            # reclassifying them as above the high threshold.
            pixels[reached & arrays["weak"]] = (248, 250, 252)
            image.pixel_array[:, :, :3] = pixels
            self.color_grid(grid, pixels[connected_crop])
            self.color_grid(isolated_grid, pixels[isolated_crop])

        self.play(UpdateFromAlphaFunc(image, propagate), run_time=2.2, rate_func=linear)
        self.play(
            Transform(note, VGroup(
                Text("connected → KEEP", font_size=22, color=EDGE).move_to([2.0, -0.95, 0]),
                Text("isolated → REJECT", font_size=22, color=WEAK).move_to([5.1, -0.95, 0]),
            )), run_time=0.3,
        )
        accepted_pixels = threshold_pixels.copy()
        accepted_pixels[arrays["weak"] & arrays["edges"]] = (248, 250, 252)
        rejected = arrays["weak"] & ~arrays["edges"]
        accepted_pixels[rejected] = (61, 70, 92)
        self.play(Transform(image, self.new_image(accepted_pixels)),
                  Transform(grid, self.pixel_grid(accepted_pixels[connected_crop], 0.28, connected_position)),
                  Transform(isolated_grid, self.pixel_grid(accepted_pixels[isolated_crop], 0.28, isolated_position)),
                  run_time=0.5)
        self.wait(1.1)
        accepted_pixels[rejected] = (16, 22, 38)
        self.play(Transform(image, self.new_image(accepted_pixels)),
                  Transform(grid, self.pixel_grid(accepted_pixels[connected_crop], 0.28, connected_position)),
                  Transform(isolated_grid, self.pixel_grid(accepted_pixels[isolated_crop], 0.28, isolated_position)),
                  run_time=0.4)
        self.wait(0.5)

        final_pixels = grayscale_rgb(arrays["edges"].astype(float))
        self.play(
            Transform(image, self.new_image(final_pixels)),
            Transform(grid, self.pixel_grid(final_pixels[connected_crop], 0.28, connected_position)),
            Transform(isolated_grid, self.pixel_grid(final_pixels[isolated_crop], 0.28, isolated_position)),
            Transform(stage_label, self.new_stage_label("5 · Final edge map", EDGE, border)),
            Transform(heading, self.side_title("Only supported edges remain", EDGE)),
            Transform(formula, Text("smooth → orient → thin → link", font_size=26, color=EDGE).move_to([3.55, -2.35, 0])),
            locator.animate.set_color(EDGE),
            second_locator.animate.set_color(REJECTED),
            dots[5].animate.set_fill(EDGE, 1), run_time=0.9,
        )
        self.wait(2.6)

    def pipeline_overview(self) -> VGroup:
        items = VGroup()
        for index, (name, color) in enumerate(zip(self.stage_names, self.stage_colors)):
            box = RoundedRectangle(
                width=2.05,
                height=0.95,
                corner_radius=0.16,
                stroke_color=color,
                stroke_width=2,
                fill_color=PANEL,
                fill_opacity=1,
            )
            label = Text(name, font_size=23, color=color, weight="BOLD")
            node = VGroup(box, label)
            items.add(node)
            if index < len(self.stage_names) - 1:
                items.add(Arrow(LEFT, RIGHT, buff=0, color=MUTED, stroke_width=2).scale(0.28))
        items.arrange(RIGHT, buff=0.18).scale_to_fit_width(14.6).move_to(DOWN * 0.1)
        return items

    def progress_rail(self) -> tuple[VGroup, list[Dot]]:
        rail = VGroup()
        dots: list[Dot] = []
        x_positions = np.linspace(-6.5, 6.5, len(self.stage_names))
        rail.add(Line([x_positions[0], -3.82, 0], [x_positions[-1], -3.82, 0], color=REJECTED))
        for x, name, color in zip(x_positions, self.stage_names, self.stage_colors):
            dot = Dot([x, -3.82, 0], radius=0.095, color=color, fill_opacity=0.15)
            label = Text(name, font_size=19, color=MUTED).next_to(dot, DOWN, buff=0.09)
            dots.append(dot)
            rail.add(dot, label)
        return rail, dots

    def image_panel(self, pixels: np.ndarray) -> tuple[ImageMobject, RoundedRectangle]:
        border = RoundedRectangle(
            width=6.0,
            height=5.55,
            corner_radius=0.14,
            stroke_color="#33405D",
            stroke_width=2,
            fill_color=PANEL,
            fill_opacity=1,
        ).move_to(LEFT * 3.55 + DOWN * 0.12)
        image = self.new_image(pixels)
        return image, border

    @staticmethod
    def new_image(pixels: np.ndarray) -> ImageMobject:
        image = ImageMobject(pixels).set_height(5.15).move_to(LEFT * 3.55 + DOWN * 0.12)
        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        return image

    @staticmethod
    def new_stage_label(text: str, color: str, border: RoundedRectangle) -> Text:
        return Text(text, font_size=28, weight="BOLD", color=color).next_to(
            border, UP, buff=0.18
        ).align_to(border, LEFT)

    @staticmethod
    def side_title(text: str, color: str) -> Text:
        return Text(text, font_size=27, weight="BOLD", color=color).move_to([3.55, 1.85, 0])

    @staticmethod
    def patch_slice(center: tuple[int, int]) -> tuple[slice, slice]:
        y, x = center
        return slice(y - 3, y + 4), slice(x - 3, x + 4)

    @staticmethod
    def pixel_grid(pixels: np.ndarray, step: float = 0.34,
                   position: np.ndarray | None = None) -> VGroup:
        if position is None:
            position = np.array([3.55, 0.35, 0])
        cells = VGroup()
        for y in range(7):
            for x in range(7):
                rgb = pixels[y, x]
                color = "#" + "".join(f"{int(v):02x}" for v in rgb)
                cells.add(Square(step, stroke_color=REJECTED, stroke_width=0.5,
                                 fill_color=color, fill_opacity=1).move_to(
                                     position + [(x - 3) * step, (3 - y) * step, 0]))
        return cells

    @staticmethod
    def color_grid(grid: VGroup, pixels: np.ndarray) -> None:
        for cell, rgb in zip(grid, pixels.reshape(-1, 3)):
            cell.set_fill("#" + "".join(f"{int(v):02x}" for v in rgb), opacity=1)

    @staticmethod
    def patch_locator(center: tuple[int, int], color: str) -> Rectangle:
        y, x = center
        scale = 5.15 / 224
        return Rectangle(width=7 * scale, height=7 * scale, color=color,
                         stroke_width=2).move_to(
                             [-3.55 + (x - 111.5) * scale,
                              -0.12 - (y - 111.5) * scale, 0])

    @staticmethod
    def gradient_step(direction: float) -> tuple[int, int]:
        angle = np.rad2deg(direction) % 180
        if angle < 22.5 or angle >= 157.5:
            return 0, 1
        if angle < 67.5:
            return 1, 1
        if angle < 112.5:
            return 1, 0
        return 1, -1

    def nms_comparison(self, arrays: dict, point: tuple[int, int]) -> VGroup:
        y, x = point
        dy, dx = self.gradient_step(arrays["direction"][point])
        coords = ((y - dy, x - dx), point, (y + dy, x + dx))
        result = VGroup()
        for index, (name, coord) in enumerate(zip(("q1", "p", "q2"), coords)):
            value = float(arrays["magnitude"][coord])
            color = STRONG if name == "p" else GRADIENT
            xpos = 2.05 + index * 1.5
            bar = Rectangle(width=0.48, height=max(0.015, value * 1.25),
                            stroke_width=0, fill_color=color, fill_opacity=1)
            bar.move_to([xpos, -1.42 + bar.height / 2, 0])
            label = Text(f"{name}: {value:.3f}", font_size=22, color=color).move_to([xpos, -1.75, 0])
            result.add(bar, label)
        return result

    def neighbor_marks(self, arrays: dict, point: tuple[int, int],
                       crop_center: tuple[int, int]) -> VGroup:
        y, x = point
        dy, dx = self.gradient_step(arrays["direction"][point])
        result = VGroup()
        for name, (py, px) in zip(("q1", "p", "q2"), ((y-dy, x-dx), point, (y+dy, x+dx))):
            position = np.array([3.55 + (px-crop_center[1])*0.34,
                                 0.35 - (py-crop_center[0])*0.34, 0])
            color = STRONG if name == "p" else GRADIENT
            box = Square(0.34, stroke_color=color, stroke_width=2.5).move_to(position)
            label = Text(name, font_size=20, color=color).move_to(position + UP * 0.48)
            result.add(box, label)
        return result

    @staticmethod
    def propagation_distances(arrays: dict) -> np.ndarray:
        """BFS depths for animation only; assert the original result is unchanged."""
        distances = np.full(arrays["strong"].shape, -1, dtype=int)
        distances[arrays["strong"]] = 0
        queue = deque(map(tuple, np.argwhere(arrays["strong"])))
        height, width = distances.shape
        while queue:
            y, x = queue.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height and 0 <= nx < width
                            and arrays["weak"][ny, nx] and distances[ny, nx] < 0):
                        distances[ny, nx] = distances[y, x] + 1
                        queue.append((ny, nx))
        assert np.array_equal(distances >= 0, arrays["edges"])
        return distances

    @staticmethod
    def hysteresis_legend() -> VGroup:
        return VGroup(
            Text("strong ≥ 0.35", font_size=22, color=STRONG),
            Text("weak 0.10–0.35", font_size=22, color=WEAK),
            Text("rejected / noise", font_size=22, color=MUTED),
        ).arrange(DOWN, buff=0.13).move_to([3.55, -2.10, 0])
