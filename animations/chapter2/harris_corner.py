"""Continuous Harris corner teaching animation backed by the Chapter 2 engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import (
    Arrow, Circle, Create, Dot, FadeIn, FadeOut, GrowArrow, Indicate, Line,
    Rectangle, ReplacementTransform, Scene, Transform, TransformFromCopy,
    Group as VGroup,
)

from chapter2_common import (
    BG, CYAN, GREEN, GRID, MUTED, ORANGE, PURPLE, WHITE, YELLOW, TextSwap,
    check_scene_bounds, framed_image, image_mobject, pixel_grid, progress_bar, text,
)
from utils import harris


@dataclass(frozen=True)
class HarrisFixture:
    images: tuple[np.ndarray, np.ndarray, np.ndarray]
    names: tuple[str, str, str]
    pipelines: tuple[harris.Pipeline, harris.Pipeline, harris.Pipeline]
    center: tuple[int, int]
    energy: np.ndarray
    threshold: float
    candidates: tuple[tuple[int, int], ...]
    keep_mask: np.ndarray
    keep_example: tuple[int, int]
    reject_example: tuple[int, int]


def build_harris_fixture() -> HarrisFixture:
    presets = harris.presets(41)
    indices = (0, 1, 3)  # flat, horizontal edge, L corner
    images = tuple(presets[harris.PRESET_NAMES[i]] for i in indices)
    pipes = tuple(harris.pipeline(image, size=5, sigma=1.2, k=.05) for image in images)
    response = pipes[2].harris
    threshold = max(0.0, float(response.max()) * .08)
    keep = harris.threshold_nms(response, threshold, 5)
    candidates = tuple(map(tuple, np.argwhere(response > threshold)))
    rejected = sorted((rc for rc in candidates if not keep[rc]),
                      key=lambda rc: -float(response[rc]))
    retained = list(map(tuple, np.argwhere(keep)))
    assert retained and rejected
    return HarrisFixture(
        images, ("Flat", "Horizontal edge", "L-corner"), pipes, (20, 20),
        harris.energy_grid(images[2], 20, 20, 9, 2), threshold, candidates, keep,
        retained[0], rejected[0],
    )


def image_point(image_mob, shape, row, col):
    h, w = shape
    x = image_mob.get_left()[0] + (col + .5) * image_mob.width / w
    y = image_mob.get_top()[1] - (row + .5) * image_mob.height / h
    return np.array([x, y, 0.0])


class HarrisCornerDetection(Scene):
    def play(self, *animations, **kwargs):
        super().play(*animations, **kwargs)
        check_scene_bounds(self)

    def cue(self, message, color=WHITE):
        new = text(message, (0, -3.43), color, 27)
        old = self.caption
        self.caption = new
        return TextSwap(old, new)

    def set_caption(self, message, color=WHITE):
        self.caption = text(message, (0, -3.43), color, 27)

    def step(self, index):
        return self.marker.animate.move_to([self.step_x[index], -4.29, 0])

    def construct(self):
        self.data = build_harris_fixture()
        self.title = text("Harris · corner from two-direction change", (0, 3.92), WHITE, 35)
        self.set_caption("Which patch can be localized in both directions?")
        self.steps, self.step_x = progress_bar(
            ["Patches", "Shift E", "Tensor", "Eigenvalues", "Response", "NMS", "Corners"])
        self.marker = Dot([self.step_x[0], -4.29, 0], radius=.045, color=CYAN)
        self.add(self.title, self.caption, self.steps, self.marker)
        self.why_corners()
        self.displacement_energy()
        self.structure_tensor()
        self.eigen_intuition()
        self.harris_response()
        self.threshold_and_nms()
        self.final_result()

    def why_corners(self):
        centers = [(-5.0, .55, 0), (0, .55, 0), (5.0, .55, 0)]
        self.patch_groups = VGroup(*[
            framed_image(image, name, center, width=3.7, accent=CYAN)
            for image, name, center in zip(self.data.images, self.data.names, centers)
        ])
        self.play(FadeIn(self.patch_groups), run_time=.8)
        self.windows = VGroup()
        self.arrows = VGroup()
        for group in self.patch_groups:
            window = Rectangle(width=.92, height=.92, color=YELLOW, stroke_width=3).move_to(group[0])
            horizontal = Arrow(window.get_center()+[-.75,-1.35,0], window.get_center()+[.75,-1.35,0],
                               buff=0, color=ORANGE, stroke_width=3)
            vertical = Arrow(window.get_center()+[1.35,.7,0], window.get_center()+[1.35,-.7,0],
                             buff=0, color=PURPLE, stroke_width=3)
            self.windows.add(window); self.arrows.add(VGroup(horizontal, vertical))
        self.play(*[Create(w) for w in self.windows], *[GrowArrow(a[0]) for a in self.arrows], run_time=.65)
        self.play(*[w.animate.shift([.32,0,0]) for w in self.windows], run_time=.45)
        self.play(*[w.animate.shift([-.32,-.32,0]) for w in self.windows],
                  *[GrowArrow(a[1]) for a in self.arrows], run_time=.5)
        labels = VGroup(
            text("little change", (-5,-2.0), MUTED, 25),
            text("one direction", (0,-2.0), ORANGE, 25),
            text("two directions", (5,-2.0), GREEN, 25),
        )
        self.play(FadeIn(labels), self.cue("Corner: intensity changes for horizontal and vertical shifts", GREEN), run_time=.6)
        self.labels = labels
        self.wait(4.05)

    def displacement_energy(self):
        corner = self.patch_groups[2]
        target = framed_image(self.data.images[2], "Same L-corner", (-4.7,.45,0), width=4.4, accent=CYAN)
        self.play(FadeOut(self.patch_groups[0]), FadeOut(self.patch_groups[1]),
                  FadeOut(self.windows), FadeOut(self.arrows), FadeOut(self.labels),
                  Transform(corner, target), self.step(1),
                  self.cue("Compare one window with a displaced copy"), run_time=.8)
        self.corner_group = corner
        base = Rectangle(width=1.0, height=1.0, color=CYAN, stroke_width=3).move_to(corner[0])
        moved = base.copy().set_color(YELLOW)
        pixel_step = corner[0].width / self.data.images[2].shape[1]
        uv = Arrow(base.get_center(), base.get_center()+[pixel_step,-pixel_step,0],
                   buff=0, tip_length=.06, color=YELLOW, stroke_width=4)
        formula = text("E(u,v) = Σ [ I(x+u,y+v) − I(x,y) ]²", (3.25,2.25), WHITE, 29)
        self.play(Create(base), TransformFromCopy(base, moved), GrowArrow(uv), FadeIn(formula), run_time=.7)
        examples = [((1,0), self.data.energy[2,3]), ((0,1), self.data.energy[3,2]),
                    ((1,1), self.data.energy[3,3]), ((-1,-1), self.data.energy[1,1])]
        self.energy_label = text("", (3.25,1.35), YELLOW, 29)
        self.add(self.energy_label)
        for (u,v), value in examples:
            new = text(f"(u,v)=({u},{v})   E={value:.0f}", (3.25,1.35), YELLOW, 29)
            shift = np.array([pixel_step*u,-pixel_step*v,0.0])
            self.play(moved.animate.move_to(base.get_center()+shift),
                      Transform(uv, Arrow(base.get_center(),base.get_center()+shift,
                                          buff=0,tip_length=.06,color=YELLOW)),
                      TextSwap(self.energy_label,new), run_time=.48)
            self.energy_label = new
            self.wait(.25)
        before,after=harris.shifted_patches(self.data.images[2],20,20,9,-1,-1)
        difference=(after-before)[3:6,3:6]
        difference_grid=pixel_grid(difference,[2.2,-.2,0],cell=.55,decimals=0,
                                   signed=True,font_size=22)
        squared_grid=pixel_grid(difference**2,[5.1,-.2,0],cell=.55,decimals=0,
                                font_size=22,accent=ORANGE)
        difference_title=text("difference · 3×3 crop",(2.2,1.0),CYAN,22)
        squared_title=text("squared",(5.1,1.0),ORANGE,22)
        sum_note=text(f"Σ over full 9×9 window = {self.data.energy[1,1]:.0f}",
                      (3.65,-1.6),YELLOW,24)
        self.play(FadeOut(self.energy_label),FadeIn(difference_grid),FadeIn(squared_grid),
                  FadeIn(difference_title),FadeIn(squared_title),FadeIn(sum_note),
                  self.cue("Subtract pixels, square differences, then sum all 81 terms"),run_time=.7)
        self.wait(.9)
        energy_grid = pixel_grid(self.data.energy, [3.25,-.65,0], cell=.58, decimals=0)
        energy_title = text("Measured E over displacements", (3.25,.95), ORANGE, 25)
        zero = Rectangle(width=.58,height=.58,color=GREEN,stroke_width=3).move_to(energy_grid[12])
        self.play(FadeOut(VGroup(difference_grid,squared_grid,difference_title,
                                 squared_title,sum_note)),
                  FadeIn(energy_grid), FadeIn(energy_title), Create(zero),
                  self.cue("The corner produces a bowl: E grows away from zero"), run_time=.75)
        self.wait(5.0)
        self.energy_objects = VGroup(base,moved,uv,formula,energy_grid,energy_title,zero)

    def structure_tensor(self):
        p = self.data.pipelines[2]
        crop = np.s_[15:26,15:26]
        gx = p.ix[crop]; gy = p.iy[crop]
        panels = VGroup(
            framed_image(gx, "Ix", (-4.8,.55,0), width=2.5, signed=True, accent=ORANGE, title_size=24),
            framed_image(gy, "Iy", (-1.75,.55,0), width=2.5, signed=True, accent=PURPLE, title_size=24),
            framed_image(gx*gx, "Ix²", (1.3,.55,0), width=2.5, signed=False, accent=ORANGE, title_size=24),
            framed_image(gx*gy, "IxIy", (4.35,.55,0), width=2.5, signed=True, accent=YELLOW, title_size=24),
            framed_image(gy*gy, "Iy²", (6.65,.55,0), width=2.0, signed=False, accent=PURPLE, title_size=24),
        )
        self.play(FadeOut(self.corner_group), FadeOut(self.energy_objects), FadeIn(panels[0]), FadeIn(panels[1]),
                  self.step(2), self.cue("Sobel derivatives first: Ix and Iy"), run_time=.8)
        self.play(FadeIn(panels[2]), FadeIn(panels[3]), FadeIn(panels[4]),
                  self.cue("Then form Ix², IxIy and Iy² at every pixel"), run_time=.75)
        self.wait(.65)
        matrix = harris.tensor_at(p.maps,*self.data.center)
        matrix_text = text(
            f"M = [ {matrix[0,0]:.4f}   {matrix[0,1]:.4f} ]\n"
            f"      [ {matrix[1,0]:.4f}   {matrix[1,1]:.4f} ]",
            (3.5,-1.8), GREEN, 31)
        integration = text("Gaussian-weighted 5 × 5 integration", (-2.9,-1.75), CYAN, 27)
        arrows = VGroup(*[Arrow(panel.get_bottom(), matrix_text.get_top()+[0,-.1,0], buff=.12,
                                      color=MUTED, stroke_width=2) for panel in panels[2:]])
        self.play(FadeIn(integration), *[GrowArrow(a) for a in arrows], FadeIn(matrix_text),
                  self.cue("Local gradient products compress into one symmetric 2 × 2 tensor", GREEN), run_time=.85)
        self.wait(5.0)
        self.tensor_objects = VGroup(panels,integration,arrows,matrix_text)

    def eigen_intuition(self):
        names = ("Flat", "Edge", "Corner")
        colors = (MUTED, ORANGE, GREEN)
        centers = (-4.7,0,4.7)
        groups = VGroup()
        for name,color,x,p in zip(names,colors,centers,self.data.pipelines):
            M = harris.tensor_at(p.maps,*self.data.center)
            large,small = np.linalg.eigvalsh(M)[::-1]
            scale = 13.0
            axes = VGroup(Line([x-1.15,-1.2,0],[x+1.15,-1.2,0],color=GRID),
                          Line([x-1.15,-1.2,0],[x-1.15,1.2,0],color=GRID))
            bars = VGroup(
                Rectangle(width=.55,height=max(.02,scale*large),fill_color=color,fill_opacity=.85,stroke_width=0)
                .move_to([x-.48,-1.2+max(.02,scale*large)/2,0]),
                Rectangle(width=.55,height=max(.02,scale*small),fill_color=CYAN,fill_opacity=.85,stroke_width=0)
                .move_to([x+.48,-1.2+max(.02,scale*small)/2,0]))
            values = text(f"λ₁={large:.3f}   λ₂={small:.3f}", (x,-1.75), WHITE, 23)
            groups.add(VGroup(text(name,(x,1.85),color,28),axes,bars,values))
        self.play(FadeOut(self.tensor_objects), FadeIn(groups), self.step(3),
                  self.cue("Eigenvalues measure change along two principal directions"), run_time=.8)
        notes = VGroup(text("both small",(-4.7,-2.35),MUTED,23),
                       text("one large",(0,-2.35),ORANGE,23),
                       text("both large",(4.7,-2.35),GREEN,23))
        self.play(FadeIn(notes), Indicate(groups[2][2],color=GREEN),
                  self.cue("Only the corner is strongly localized in both directions", GREEN), run_time=.65)
        self.wait(5.0)
        self.eigen_objects = VGroup(groups,notes)

    def harris_response(self):
        p = self.data.pipelines[2]
        M = harris.tensor_at(p.maps,*self.data.center)
        det,trace = float(np.linalg.det(M)),float(np.trace(M))
        response = float(p.harris[self.data.center])
        heat = framed_image(p.harris,"Signed Harris response",(-3.7,.25,0),width=5.2,signed=True,accent=ORANGE)
        formula = text("R = det(M) − k · trace(M)²", (3.35,1.7), WHITE, 31)
        values = text(f"k = 0.05\ndet = {det:.5f}\ntrace = {trace:.5f}\nR = {response:.5f}",
                      (3.35,.2), YELLOW, 28)
        legend = VGroup(text("positive: corner",(3.35,-1.55),ORANGE,24),
                        text("negative: edge",(3.35,-2.0),CYAN,24))
        self.play(FadeOut(self.eigen_objects),FadeIn(heat),FadeIn(formula),self.step(4),
                  self.cue("Harris combines both eigen-directions into one signed score"),run_time=.8)
        self.play(FadeIn(values),FadeIn(legend),run_time=.65)
        self.wait(4.0)
        self.response_objects=VGroup(heat,formula,values,legend)

    def threshold_and_nms(self):
        p=self.data.pipelines[2]
        heat=self.response_objects[0]
        self.play(FadeOut(VGroup(*self.response_objects[1:])),heat.animate.move_to([-4.7,.25,0]),self.step(5),
                  self.cue("Threshold gives candidates; 5 × 5 NMS keeps local peaks"),run_time=.7)
        dots=VGroup(*[Dot(image_point(heat[0],p.harris.shape,*rc),radius=.035,color=YELLOW)
                      for rc in self.data.candidates])
        threshold=text(f"threshold = {self.data.threshold:.6f}",(-4.7,-2.65),YELLOW,23)
        units=text("5 × 5 values shown as 1000 × R",(3.2,2.25),MUTED,19)
        self.play(FadeIn(dots),FadeIn(threshold),FadeIn(units),run_time=.55)
        self.nms_label=text("",(3.2,2.65),WHITE,27); self.add(self.nms_label)
        self.nms_grid=None
        for rc,color in ((self.data.keep_example,GREEN),(self.data.reject_example,ORANGE)):
            r,c=rc; patch=1000*p.harris[r-2:r+3,c-2:c+3]
            grid=pixel_grid(patch,[3.2,.25,0],cell=.72,decimals=2,signed=True,font_size=17)
            center=Rectangle(width=.72,height=.72,color=YELLOW,stroke_width=3).move_to(grid[12])
            decision,reason=harris.nms_decision(p.harris,self.data.threshold,5,r,c)
            label=text(f"p=[{r},{c}]  {'KEEP' if decision else 'REJECT'}",(3.2,2.65),color,28)
            if self.nms_grid is None:
                self.play(FadeIn(grid),Create(center),TextSwap(self.nms_label,label),run_time=.6)
            else:
                self.play(FadeOut(self.nms_grid),FadeOut(self.nms_center),FadeIn(grid),Create(center),
                          TextSwap(self.nms_label,label),run_time=.65)
            self.nms_label=label; self.nms_grid=grid; self.nms_center=center
            explanation=("KEEP · maximum in the 5 × 5 window" if decision else
                         "REJECT · neighbor has a larger response")
            self.play(Indicate(grid[12],color=color),self.cue(explanation,color),run_time=.55)
            self.wait(2.2)
        kept=VGroup(*[Circle(radius=.12,color=GREEN,stroke_width=3).move_to(
            image_point(heat[0],p.harris.shape,*rc)) for rc in map(tuple,np.argwhere(self.data.keep_mask))])
        self.play(FadeOut(dots),FadeIn(kept),self.cue("Rejected responses fade; the local maximum remains",GREEN),run_time=.65)
        self.wait(2.9)
        self.nms_objects=VGroup(heat,threshold,units,self.nms_grid,self.nms_center,self.nms_label,kept)

    def final_result(self):
        original=framed_image(self.data.images[2],"Detected on original",(0,.45,0),width=5.2,accent=CYAN)
        markers=VGroup(*[Circle(radius=.16,color=GREEN,stroke_width=4).move_to(
            image_point(original[0],self.data.images[2].shape,*rc))
            for rc in map(tuple,np.argwhere(self.data.keep_mask))])
        self.play(FadeOut(self.nms_objects),FadeIn(original),FadeIn(markers),self.step(6),
                  self.cue("Harris detects locations stable under small 2D shifts",GREEN),run_time=.8)
        self.wait(1.1)
        self.play(self.cue("Single-scale Harris does not guarantee scale invariance",PURPLE),run_time=.45)
        self.wait(.65)
        self.play(self.cue("Next: SIFT adds scale, orientation and description",PURPLE),run_time=.55)
        self.wait(4.5)
