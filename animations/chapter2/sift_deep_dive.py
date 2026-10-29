"""SIFT Deep Dive V2: one computed camera image through every SIFT stage.

Render from animations/chapter2 with the existing root .venv. All numbers are
precomputed once by sift_deep_dive_data; there are no numerical updaters.
"""

from __future__ import annotations

from collections import Counter
from math import cos, radians, sin

import cv2
import numpy as np
from manim import (Arrow, Circle, Create, Dot, FadeIn, FadeOut, Group as VGroup,
                   Indicate, Line, Rectangle, Scene, Text, Transform)

from chapter2_common import (BG, CYAN, GREEN, GRID, MUTED, ORANGE, PURPLE,
                             WHITE, YELLOW, check_scene_bounds, image_mobject,
                             pixel_grid)
from sift_deep_dive_data import (build_data, dog_examples,
                                 gaussian_pixel_example)
from utils import scale_space, sift


def text(value, xy=(0,0), color=WHITE, size=28):
    """Arial contains the Vietnamese glyphs missing from the default font."""
    return Text(value, font="Arial", font_size=size, color=color).move_to(
        [xy[0],xy[1],0])


def image_point(mob, shape, row, col):
    return np.array([mob.get_left()[0] + (col + .5) * mob.width / shape[1],
                     mob.get_top()[1] - (row + .5) * mob.height / shape[0], 0])


def bars(values, center=(0, 0), width=7, height=2.25, color=PURPLE):
    a = np.asarray(values, dtype=float).ravel()
    peak = max(float(np.max(a)), 1e-12)
    result = VGroup()
    for i, value in enumerate(a):
        w = width / len(a) * .78
        h = max(.008, float(value) / peak * height)
        x = center[0] + (i + .5) * width / len(a) - width / 2
        y = center[1] - height / 2 + h / 2
        result.add(Rectangle(width=w, height=h, stroke_width=0,
                             fill_color=color, fill_opacity=.9).move_to([x, y, 0]))
    return result


def vector_map(values, center=(0, 0), cell=.30, color=ORANGE):
    a = np.asarray(values).reshape(8, 16)
    peak = max(float(np.max(a)), 1e-12)
    result = VGroup()
    for r in range(8):
        for c in range(16):
            p = [center[0] + (c - 7.5) * cell,
                 center[1] + (3.5 - r) * cell, 0]
            result.add(Rectangle(width=cell * .92, height=cell * .92,
                                 stroke_color=GRID, stroke_width=.6,
                                 fill_color=color,
                                 fill_opacity=.06 + .86 * float(a[r, c]) / peak).move_to(p))
    return result


class SIFTDeepDive(Scene):
    """A computed, ten-part lecture from Gaussian pixels to matched descriptors."""

    def play(self, *animations, **kwargs):
        super().play(*animations, **kwargs)
        try:
            check_scene_bounds(self)
        except AssertionError as exc:
            for root in self.mobjects:
                for mob in root.get_family():
                    if mob.has_points() and (mob.get_left()[0] <= -8.02 or
                            mob.get_right()[0] >= 8.02 or
                            mob.get_bottom()[1] <= -4.52 or
                            mob.get_top()[1] >= 4.52):
                        raise AssertionError(f"Out of frame: {type(mob).__name__} "
                                             f"{mob.get_bounding_box()}") from exc
            raise

    def section(self, heading, caption):
        self.storyboard.append((heading, float(self.renderer.time)))
        persistent=(self.heading,self.caption,self.progress,self.marker,self.hero)
        extras=[mob for mob in self.mobjects if all(mob is not item for item in persistent)]
        if extras:
            self.play(*[FadeOut(item) for item in extras], run_time=.22)
            self.remove(*extras)
        self.details = VGroup()
        self.play(FadeOut(self.heading), FadeOut(self.caption), run_time=.18)
        self.heading = text(heading, (0, 3.91), WHITE, 33)
        self.caption = text(caption, (0, -3.42), WHITE, 26)
        self.play(FadeIn(self.heading), FadeIn(self.caption), run_time=.25)

    def add_detail(self, *objects, run_time=.5):
        group = VGroup(*objects)
        self.play(FadeIn(group), run_time=run_time)
        self.details.add(group)
        return group

    def hero_to(self, values, center=(-4.7, .15), width=4.4, run_time=.75,
                signed=False):
        display = np.asarray(values)
        if display.shape != self.hero.pixel_array.shape[:2]:
            # Pixel-preserving display enlargement keeps ImageMobject shape
            # stable for Transform. The source array retains its true dimensions.
            h,w = self.hero.pixel_array.shape[:2]
            display = cv2.resize(display,(w,h),interpolation=cv2.INTER_NEAREST)
        target = image_mobject(display, width=width, signed=signed).move_to(
            [center[0],center[1],0])
        self.play(Transform(self.hero, target), run_time=run_time)

    def hold(self, seconds=3):
        self.wait(seconds)

    def construct(self):
        self.data = build_data()
        self.storyboard = []
        self.heading = text("SIFT Deep Dive V2", (0, 3.91), WHITE, 33)
        self.caption = text("Ảnh gốc → đặc trưng có thể so khớp", (0, -3.42), WHITE, 26)
        names=["Ảnh", "Gaussian", "Octave", "DoG", "Cực trị", "Tinh chỉnh",
               "Hướng", "128D", "So khớp", "Tóm tắt"]
        self.progress_x=np.linspace(-6.6,6.6,len(names))
        self.progress=VGroup(*[text(name,(float(x),-4.15),MUTED,18)
                               for x,name in zip(self.progress_x,names)])
        self.marker = Dot([self.progress_x[0], -4.43, 0], radius=.045, color=CYAN)
        self.hero = image_mobject(self.data.image, width=4.4).move_to([-4.7, .15, 0])
        self.details = VGroup()
        self.add(self.heading, self.caption, self.progress, self.marker, self.hero)
        self.opening()
        self.gaussian_pixel()
        self.increasing_sigma()
        self.gaussian_pyramid()
        self.dog_pixel()
        self.dog_stacks()
        self.dog_usefulness()
        self.extrema()
        self.localization()
        self.filtering()
        self.orientation()
        self.descriptor()
        self.normalization()
        self.matching()
        self.recap()

    def step(self, index):
        self.play(self.marker.animate.move_to([self.progress_x[index], -4.43, 0]),
                  run_time=.3)

    def opening(self):
        self.section("Một vật thể, nhiều cách nhìn",
                     "Máy tính nhận ra cùng vật thể khi xoay và đổi kích thước thế nào?")
        self.add_detail(text("Máy ảnh tổng hợp · 129 × 129 pixel", (-4.7, 2.7), CYAN, 23),
                        text("Cùng vật thể · xoay 17° · tỷ lệ 0.94", (3.1, 2.7), PURPLE, 23))
        other = image_mobject(self.data.transformed, width=4.4).move_to([3.1, .15, 0])
        self.add_detail(other)
        self.hold(9)

    def gaussian_pixel(self):
        self.section("Gaussian: một pixel được tính thế nào?",
                     "Nhân cường độ với trọng số rồi cộng toàn bộ lân cận")
        self.step(1)
        patch, kernel, total = gaussian_pixel_example(self.data)
        r, c = 50, 35
        crop = self.data.image[r-3:r+4, c-3:c+4]
        neighborhood = pixel_grid(crop, (1.75, .65, 0), cell=.46, decimals=2, font_size=15)
        weights = pixel_grid(kernel[6:9, 6:9], (5.4, .75, 0), cell=.64,
                             decimals=3, font_size=15, accent=PURPLE)
        dot = Circle(radius=.11, color=YELLOW).move_to(image_point(self.hero,
                     self.data.image.shape, r, c))
        self.add_detail(dot, neighborhood, weights,
                        text("7×7 ảnh thật", (1.75, 2.55), CYAN, 22),
                        text("tâm kernel 15×15", (5.4, 2.55), PURPLE, 22),
                        text(f"I[{r},{c}] = {self.data.image[r,c]:.3f}",
                             (1.7, -1.55), YELLOW, 23),
                        text(f"{patch[7,7]:.3f} × {kernel[7,7]:.4f} = "
                             f"{patch[7,7]*kernel[7,7]:.5f}",
                             (5.1, -1.55), ORANGE, 22))
        self.hold(4)
        result = text(f"Σ 225 tích = L[{r},{c},1.6] = {total:.5f}",
                      (2.7, -2.43), GREEN, 25)
        self.add_detail(result)
        self.play(Indicate(neighborhood[24], color=YELLOW), run_time=.55)
        self.play(dot.animate.move_to(image_point(self.hero, self.data.image.shape,
                                                  r, c+3)), run_time=.45)
        self.hero_to(self.data.pyramid[0].gaussians[0])
        self.add_detail(text("L(x,y,σ) = G(x,y,σ) * I(x,y)", (0, 3.11), WHITE, 25))
        self.hold(10)

    def increasing_sigma(self):
        self.section("Tăng σ: chi tiết nhỏ mờ trước",
                     "Mỗi ảnh bên phải là mảng Gaussian tính từ cùng máy ảnh")
        self.hero_to(self.data.image)
        p = self.data.pyramid[0]
        xs = (-.8, 1.5, 3.8, 6.1)
        thumbnails = VGroup()
        for j, x in zip((0, 1, 3, 5), xs):
            im = image_mobject(p.gaussians[j], width=1.8).move_to([x, .55, 0])
            detail = image_mobject(p.gaussians[j][40:57,91:108],
                                   width=1.3).move_to([x,-1.6,0])
            label = text(f"σ={p.local_sigmas[j]:.2f}", (x, 2.0), PURPLE, 19)
            thumbnails.add(im,detail,label)
            self.add_detail(im,detail,label, run_time=.65)
            self.hold(1.8)
        self.add_detail(text("Cùng nút chụp, cùng tọa độ: chi tiết nhỏ dần mờ",
                             (2.55, -2.68), YELLOW, 21))
        self.hold(8)

    def gaussian_pyramid(self):
        self.section("Gaussian pyramid: 3 octave × 6 tầng",
                     "Trong mỗi octave: làm mờ tăng dần; giữa octave: lấy mẫu cách 2 pixel")
        self.step(2)
        self.hero_to(self.data.image, center=(5.5, .35), width=2.6)
        k = 2 ** (1 / 3)
        k_note=text(f"σ0 = 1.6     k = 2^(1/3) = {k:.4f}",
                    (2.35, 2.95), PURPLE, 24)
        self.add_detail(k_note)
        all_tiles = VGroup()
        for o, octave in enumerate(self.data.pyramid):
            if o:
                previous = self.data.pyramid[o-1]
                self.hero_to(previous.gaussians[3], center=(5.5,.35), width=2.6, run_time=.55)
                grid = pixel_grid(previous.gaussians[3][46:50, 46:50],
                                  (5.5, -1.65, 0), cell=.42, decimals=2, font_size=13)
                sampled = VGroup(*[Circle(radius=.16,color=YELLOW,stroke_width=2)
                                   .move_to(grid[i]) for i in (0,2,8,10)])
                transition = self.add_detail(
                    grid,sampled,text(f"G({o-1},3)[::2,::2] → G({o},0)",
                               (5.5, -2.65), PURPLE, 19))
                self.hold(3)
                self.play(FadeOut(transition), run_time=.2)
                self.remove(transition)
                self.details.remove(transition)
                self.hero_to(octave.gaussians[0], center=(5.5,.35), width=2.6)
            else:
                self.hero_to(octave.gaussians[0], center=(5.5,.35), width=2.6)
            dims = octave.gaussians[0].shape[0]
            row_y = 1.95 - o*1.75
            row = VGroup()
            self.add_detail(text(f"Octave {o} · {dims}²", (-6.1, row_y+.82), PURPLE, 19))
            for j in range(6):
                x = -4.8 + j*1.65
                tile = image_mobject(octave.gaussians[j], width=1.25).move_to([x,row_y,0])
                label = text(f"G({o},{j})", (x,row_y-.77), WHITE, 17)
                info = text(f"σl={octave.local_sigmas[j]:.2f}  "
                            f"σg={octave.global_sigmas[j]:.2f}  "
                            f"Δ={octave.increments[j]:.2f}",
                            (5.5, -2.38), PURPLE, 17)
                if j:
                    self.hero_to(octave.gaussians[j], center=(5.5,.35), width=2.6, run_time=.55)
                self.play(FadeIn(tile), FadeIn(label), FadeIn(info), run_time=.5)
                self.wait(2.0)
                self.play(FadeOut(info), run_time=.15)
                row.add(tile,label)
            all_tiles.add(row)
            self.hold(3)
        self.play(FadeOut(k_note),run_time=.2)
        self.remove(k_note)
        self.add_detail(text("18 Gaussian · σ_global = σ_local × 2^octave",
                             (1.7, 3.15), GREEN, 22))
        self.hold(8)

    def dog_pixel(self):
        self.section("DoG: trừ hai tầng trong cùng octave",
                     "D(0,1) = G(0,2) − G(0,1), từng pixel cùng tọa độ")
        self.step(3)
        octave = self.data.pyramid[0]
        self.hero_to(octave.gaussians[1], center=(-5.1, .35), width=2.8)
        high = image_mobject(octave.gaussians[2], width=2.8).move_to([0,.35,0])
        diff = image_mobject(octave.dogs[1], width=2.8, signed=True).move_to([5.1,.35,0])
        labels = [text("G(0,1)", (-5.1,2.05), CYAN, 25),
                  text("G(0,2)", (0,2.05), PURPLE, 25),
                  text("D(0,1)", (5.1,2.05), ORANGE, 25)]
        self.add_detail(high, *labels)
        for r,c,lo,hi,value in dog_examples(self.data):
            a = Circle(radius=.095, color=YELLOW).move_to(image_point(self.hero,
                              octave.gaussians[1].shape,r,c))
            b = Circle(radius=.095, color=YELLOW).move_to(image_point(high,
                              octave.gaussians[2].shape,r,c))
            equation = text(f"[{r},{c}]: {hi:.4f} − {lo:.4f} = {value:+.4f}",
                            (0,-2.1), YELLOW, 24)
            self.play(FadeIn(a), FadeIn(b), FadeIn(equation), run_time=.45)
            self.hold(1.8)
            self.play(FadeOut(a), FadeOut(b), FadeOut(equation), run_time=.18)
        self.add_detail(diff, text("âm: xanh · dương: cam · 0: tối",
                                   (5.1,-1.65), WHITE, 20),
                        text("Độ tương phản chỉ đổi cách hiển thị, không đổi số DoG",
                             (0,-2.63), MUTED, 20))
        self.hold(7)

    def dog_stacks(self):
        self.section("15 ảnh DoG từ 18 ảnh Gaussian",
                     "Mỗi mũi tên ghép đúng hai tầng Gaussian liền kề")
        self.hero_to(self.data.pyramid[0].dogs[0], center=(5.5,.4), width=2.6,
                     signed=True)
        for o, octave in enumerate(self.data.pyramid):
            row_y = 1.85 - o*2.15
            self.add_detail(text(f"O{o} · {octave.dogs[0].shape[0]}²",
                                 (-6.3,row_y+.35), PURPLE, 17))
            gaussian_tiles=[]
            for j in range(6):
                x=-4.9+j*1.35
                gaussian_tiles.append(image_mobject(octave.gaussians[j],width=.62)
                                      .move_to([x,row_y+.78,0]))
            self.add_detail(*gaussian_tiles,
                            *[text(f"G({o},{j})",(-4.9+j*1.35,row_y+1.19),
                                   PURPLE,14) for j in range(6)],run_time=.6)
            for j in range(5):
                x=-4.225+j*1.35
                tile=image_mobject(octave.dogs[j],width=.93,signed=True)
                tile.move_to([x,row_y,0])
                label=text(f"D({o},{j})",(x,row_y-.59),ORANGE,15)
                parents=(Arrow(gaussian_tiles[j].get_bottom(),tile.get_center(),
                               buff=0,color=ORANGE,stroke_width=1.8,
                               max_tip_length_to_length_ratio=.22),
                         Arrow(gaussian_tiles[j+1].get_bottom(),tile.get_center(),
                               buff=0,color=ORANGE,stroke_width=1.8,
                               max_tip_length_to_length_ratio=.22))
                self.hero_to(octave.dogs[j],center=(5.5,.4),width=2.6,
                             run_time=.48,signed=True)
                self.play(FadeIn(tile),FadeIn(label),
                          *[FadeIn(line) for line in parents],run_time=.5)
                self.details.add(tile,label,*parents)
                self.hold(1.7)
            self.hold(2.3)
        self.add_detail(text("18 G → 15 DoG",(5.5,2.85),GREEN,22))
        self.hold(7)

    def dog_usefulness(self):
        self.section("Cùng một vị trí qua 5 thang DoG",
                     "Cấu trúc đáp ứng khác nhau khi mức làm mờ thay đổi")
        kp=self.data.candidate
        octave=self.data.pyramid[kp.octave]
        self.hero_to(octave.dogs[0],center=(-4.7,.2),width=3.4,signed=True)
        values=[float(d[kp.row,kp.col]) for d in octave.dogs]
        line=Line([-.2,.25,0],[6.7,.25,0],color=GRID)
        self.add_detail(line,text(f"Pixel [{kp.row},{kp.col}] · DoG có dấu",
                                  (3.2,2.65),YELLOW,22))
        dots=VGroup()
        for j,value in enumerate(values):
            x=.3+j*1.35
            y=.25+value*18
            dot=Dot([x,y,0],radius=.09,color=ORANGE if value>0 else CYAN)
            dots.add(dot)
            self.hero_to(octave.dogs[j],center=(-4.7,.2),width=3.4,
                         signed=True,run_time=.45)
            label=text(f"D({kp.octave},{j}) = {value:+.4f}",
                       (3.2,-2.2),ORANGE if value>0 else CYAN,20)
            item=self.add_detail(dot,label)
            self.hold(1.1)
            self.play(FadeOut(label),run_time=.15)
            self.remove(label)
            item.remove(label)
        for a,b in zip(dots[:-1],dots[1:]):
            self.add_detail(Line(a.get_center(),b.get_center(),color=YELLOW),run_time=.3)
        self.hold(6)

    def extrema(self):
        self.section("Cực trị trong 26 điểm lân cận",
                     "Tâm ở tầng giữa: 8 điểm cùng tầng + 9 trước + 9 sau")
        self.step(4)
        d=self.data; kp=d.candidate; octave=d.pyramid[kp.octave]
        self.hero_to(octave.dogs[kp.layer],center=(-5.75,.4),width=2.4,
                     signed=True)
        self.add_detail(text(f"p = D({kp.octave},{kp.layer})[{kp.row},{kp.col}]",
                             (-4.5,2.2),YELLOW,22))
        regions=[]
        for j,x in zip((kp.layer-1,kp.layer,kp.layer+1),(-3.35,.15,3.65)):
            block=octave.dogs[j][kp.row-1:kp.row+2,kp.col-1:kp.col+2]*1000
            grid=pixel_grid(block,(x,.25,0),cell=.66,decimals=1,signed=True,font_size=15)
            regions.append(grid)
            self.add_detail(grid,text(f"D({kp.octave},{j}) × 1000",(x,1.9),PURPLE,20))
        neighbors=scale_space.inspect_neighbors(octave.dogs,kp.layer,kp.row,kp.col)
        actual=np.array([e.value for e in neighbors])
        center=Circle(radius=.28,color=YELLOW).move_to(regions[1][4])
        self.add_detail(center,text(f"p={kp.dog_value*1000:+.2f} · "
                                    f"min láng giềng={actual.min()*1000:+.2f}",
                                    (2.6,-1.72),YELLOW,23))
        for group in (VGroup(*regions[1][:4],*regions[1][5:]),
                      regions[0],regions[2]):
            self.play(Indicate(group,color=YELLOW),run_time=.65)
            self.hold(1.0)
        self.add_detail(text(f"{kp.kind}: so sánh strict với cả 26 → ứng viên thô",
                             (1.9,-2.5),GREEN,22))
        self.hold(10)
        # One real adjacent location that fails the strict comparison.
        r,c=kp.row,kp.col+1
        invalid=scale_space.inspect_neighbors(octave.dogs,kp.layer,r,c)
        bad=float(octave.dogs[kp.layer][r,c])
        assert not (all(bad>e.value for e in invalid) or
                    all(bad<e.value for e in invalid))
        self.add_detail(text(f"[{r},{c}]: {bad*1000:+.2f} · không cực trị → loại",
                             (1.9,-2.03),MUTED,22))
        self.hold(4)

    def localization(self):
        self.section("Tinh chỉnh vị trí dưới một pixel",
                     "Taylor 3D quanh cực trị: giải Hδ = −g")
        self.step(5)
        d=self.data; raw=d.candidate; kp=d.selected
        octave=d.pyramid[raw.octave]
        self.hero_to(octave.dogs[raw.layer],center=(-4.65,.3),width=3.2,
                     signed=True)
        t=sift.calculate_dog_derivatives(octave.dogs,kp.layer,kp.row,kp.col)
        grid_center=np.array([-4.5,.25,0])
        raw_center=grid_center+np.array([.5*(raw.col-kp.col),
                                         -.5*(raw.row-kp.row),0])
        origin=Dot(raw_center,radius=.13,color=YELLOW)
        grid=VGroup(*[Line([-5.5+i*.5,-.75,0],[-5.5+i*.5,1.25,0],color=GRID)
                       for i in range(5)],
                    *[Line([-5.5,-.75+i*.5,0],[-3.5,-.75+i*.5,0],color=GRID)
                      for i in range(5)])
        g=t.gradient*1000; h=t.hessian*1000
        matrix=VGroup(*[text("  ".join(f"{v:+.1f}" for v in row),
                             (2.3,1.55-i*.48),PURPLE,22)
                        for i,row in enumerate(h)])
        self.add_detail(grid,origin,
                        text("g × 1000 = ("+", ".join(f"{v:+.2f}" for v in g)+")",
                             (2.3,2.45),CYAN,22),
                        text("H × 1000 (x,y,scale)",(2.3,1.95),PURPLE,22),matrix)
        self.hold(4)
        if (raw.col,raw.row,raw.layer)!=(kp.col,kp.row,kp.layer):
            self.play(origin.animate.move_to(grid_center),run_time=.7)
            self.add_detail(text(f"Recenter: ({raw.col},{raw.row},{raw.layer}) → "
                                 f"({kp.col},{kp.row},{kp.layer})",
                                 (2.3,-2.78),YELLOW,20))
            self.hold(2)
        self.play(origin.animate.shift([kp.delta[0]*.5,-kp.delta[1]*.5,0]),
                  run_time=.85)
        self.add_detail(text("δ = ("+", ".join(f"{v:+.3f}" for v in kp.delta)+")",
                             (2.3,-.55),YELLOW,24),
                        text(f"({kp.col},{kp.row},{kp.layer}) → "
                             f"({kp.col_local:.3f},{kp.row_local:.3f},{kp.layer_local:.3f})",
                             (2.3,-1.35),GREEN,21),
                        text(f"D_interp = {kp.response:+.5f} · σ={kp.sigma_local:.3f}",
                             (2.3,-2.15),ORANGE,24))
        self.hold(13)

    def filtering(self):
        self.section("Hai bộ lọc cho ứng viên đã tinh chỉnh",
                     "Tương phản yếu và biên kéo dài bị loại vì thiếu ổn định")
        d=self.data
        counts=Counter(p.status for p in d.filtered)
        edge=next(p for p in d.filtered if p.status=="edge")
        h=sift.calculate_spatial_hessian(d.pyramid[edge.octave].dogs,
                                          edge.layer,edge.row,edge.col)
        edge_test=sift.edge_decision(h)
        low=next(p for p in d.filtered_b if p.status=="low contrast")
        self.hero_to(d.image,center=(-4.8,.35),width=3.2)
        self.add_detail(text("A · ảnh gốc",(-4.8,2.25),CYAN,22),
                        text(f"Biên [{edge.row},{edge.col}] · 2×2 Hessian",
                             (2.2,2.25),ORANGE,23),
                        text(f"[{h[0,0]:+.3f}  {h[0,1]:+.3f}]",
                             (2.2,1.52),PURPLE,22),
                        text(f"[{h[1,0]:+.3f}  {h[1,1]:+.3f}]",
                             (2.2,1.08),PURPLE,22),
                        text(f"trace²/det = {edge_test[3]:.2f} · giới hạn 12.1",
                             (2.2,.30),ORANGE,22),
                        text("2×2 kiểm tra biên; 3×3 trước đó định vị",
                             (2.2,-.45),WHITE,20),
                        text("→ REJECT: định vị kém theo biên",(2.2,-1.12),MUTED,22))
        self.hold(8)
        self.section("Loại phản hồi tương phản thấp",
                     "Cùng vật thể sau biến đổi cũng có ứng viên quá yếu")
        self.hero_to(d.transformed,center=(-4.8,.35),width=3.2)
        self.add_detail(text("B · cùng vật thể đã xoay",(-4.8,2.25),PURPLE,22),
                        text(f"Tương phản yếu: |D_interp|={abs(low.response):.5f} < 0.008",
                             (2.2,1.2),YELLOW,23),
                        text("→ REJECT: phản hồi quá nhỏ",(2.2,.32),MUTED,22),
                        text(f"A: {len(d.raw)} thô → {counts['kept']} vị trí giữ",
                             (2.2,-1.2),GREEN,23),
                        text(f"B: {len(d.raw_b)} thô → "
                             f"{sum(p.status=='kept' for p in d.filtered_b)} vị trí giữ",
                             (2.2,-2.0),GREEN,23))
        self.hold(9)

    def orientation(self):
        self.section("Hướng chủ đạo từ gradient cục bộ",
                     "36 ô hướng; mỗi pixel bỏ phiếu theo độ lớn và khoảng cách")
        self.step(6)
        d=self.data; kp=d.selected; info=d.orientation
        image,j=sift.gaussian_level(kp,d.pyramid[kp.octave])
        self.hero_to(image,center=(-4.8,.2),width=3.5)
        gx,gy=sift.gradient_field(image)
        center=Dot(image_point(self.hero,image.shape,kp.row_local,kp.col_local),
                   radius=.11,color=YELLOW)
        arrows=VGroup()
        for row,col,mag,angle,spatial,weight in sorted(
                info.contributions,key=lambda s:-s[5])[:9]:
            start=image_point(self.hero,image.shape,row,col)
            vector=np.array([gx[row,col],-gy[row,col],0])
            norm=np.linalg.norm(vector)
            if norm>1e-9:
                arrows.add(Arrow(start,start+vector/norm*.38,buff=0,
                                 stroke_width=2.5,color=YELLOW,max_tip_length_to_length_ratio=.25))
        hist=bars(info.histogram,center=(2.25,.25),width=7.2,height=2.5,color=PURPLE)
        self.add_detail(center,arrows,text(f"Gaussian G({kp.octave},{j})",
                                            (-4.8,2.45),CYAN,22),
                        text("36 bin × 10°",(2.25,2.4),PURPLE,24))
        self.hold(4)
        self.add_detail(hist)
        for row,col,mag,angle,spatial,weight in sorted(
                info.contributions,key=lambda s:-s[5])[:3]:
            low=int(np.floor(angle/10))%36
            mark=Circle(radius=.10,color=YELLOW).move_to(
                image_point(self.hero,image.shape,row,col))
            evidence=text(f"[{row},{col}] · |∇|={mag:.3f} × w={spatial:.3f}"
                          f" → {weight:.3f} · θ={angle:.0f}°",
                          (2.15,-1.65),YELLOW,21)
            self.play(FadeIn(mark),FadeIn(evidence),Indicate(hist[low],color=YELLOW),
                      Indicate(hist[(low+1)%36],color=YELLOW),run_time=.6)
            self.hold(1.0)
            self.play(FadeOut(mark),FadeOut(evidence),run_time=.15)
        self.add_detail(text(f"Đỉnh 36 bin: {', '.join(f'{a:.0f}°' for a in info.peaks)}",
                             (2.15,-1.65),GREEN,23),
                        text("Đỉnh phụ hợp lệ ≥ 80% đỉnh cao nhất",
                             (2.15,-2.32),WHITE,21))
        self.hold(11)
        theta=radians(d.oriented.orientation)
        a=Arrow([2.0,.1,0],[2+1.4*cos(theta),.1-1.4*sin(theta),0],
                buff=0,color=YELLOW)
        self.add_detail(a,text(f"Khung descriptor xoay theo {d.oriented.orientation:.0f}°",
                               (2.15,2.95),YELLOW,22))
        self.hold(5)
        alternative=next(p for p in d.filtered if p.status=="kept" and
                         p.octave==kp.octave and p is not kp and
                         len(sift.calculate_orientation_histogram(
                             p,d.pyramid[p.octave]).peaks)==2 and
                         sift.gaussian_level(p,d.pyramid[p.octave])[1]==j)
        alt_info=sift.calculate_orientation_histogram(
            alternative,d.pyramid[alternative.octave])
        self.section("Một vị trí, hai hướng hợp lệ",
                     "Hai bản ghi hướng cùng neo tại một keypoint, không phải hai vị trí")
        anchor=image_point(self.hero,image.shape,
                           alternative.row_local,alternative.col_local)
        alt_arrows=VGroup(*[Arrow(anchor,anchor+.68*np.array(
            [cos(radians(deg)),-sin(radians(deg)),0]),buff=0,
            color=YELLOW if i==0 else PURPLE)
            for i,deg in enumerate(alt_info.peaks)])
        self.add_detail(Dot(anchor,radius=.1,color=GREEN),alt_arrows,
                        bars(alt_info.histogram,center=(2.2,.2),
                             width=7.0,height=2.3,color=PURPLE),
                        text(f"[{alternative.row},{alternative.col}] → "
                             f"{alt_info.peaks[0]:.0f}° và {alt_info.peaks[1]:.0f}°",
                             (2.2,2.65),GREEN,22))
        self.hold(7)

    def descriptor(self):
        self.section("Descriptor: 4 × 4 ô, mỗi ô 8 hướng",
                     "Mẫu gradient thật được nội suy vào ô không gian và hướng")
        self.step(7)
        d=self.data; desc=d.descriptor; kp=d.oriented
        patch,mask=sift.sample_oriented_descriptor_patch(kp,d.pyramid[kp.octave],48)
        self.hero_to(patch,center=(-4.65,.2),width=3.6)
        cells=VGroup()
        for r in range(4):
            for c in range(4):
                x=-4.65+(c-1.5)*.9; y=.2+(1.5-r)*.9
                cells.add(Rectangle(width=.9,height=.9,stroke_color=PURPLE,
                                    stroke_width=1.2,fill_opacity=0).move_to([x,y,0]))
        self.add_detail(cells,text(f"4×4 ô · mỗi ô rộng 3σ = {3*kp.sigma_local:.2f} px",
                                   (1.8,2.8),PURPLE,20),
                        text("Patch xoay từ ảnh Gaussian",
                             (-4.65,-2.45),CYAN,20))
        self.hold(5)
        sample=max(desc.samples,key=lambda s:s[5])
        row,col,u,v,angle,weight=sample
        rr=int(np.clip(round(v),0,3));cc=int(np.clip(round(u),0,3))
        bb=int(np.floor(angle/45))%8
        index=sift.descriptor_index(rr,cc,bb)
        sample_dot=Dot(image_point(self.hero,patch.shape,
                                   (v+.5)*patch.shape[0]/4,
                                   (u+.5)*patch.shape[1]/4),
                       radius=.075,color=YELLOW)
        one_cell=bars(desc.cells[rr,cc],center=(2.0,.35),width=5.5,height=2.1,color=ORANGE)
        active=cells[rr*4+cc].copy().set_stroke(YELLOW,3)
        self.add_detail(sample_dot,active,one_cell,
                        text(f"Mẫu [{row},{col}] · u={u:.2f}, v={v:.2f}",
                             (2.0,2.2),YELLOW,21),
                        text(f"θ_rel={angle:.1f}° · vote={weight:.4f}",
                             (2.0,-1.3),ORANGE,22))
        self.play(Indicate(one_cell[bb],color=YELLOW),run_time=.6)
        self.hold(6)
        self.add_detail(text(f"index=({rr}×4+{cc})×8+{bb}={index}",
                             (2.0,-1.93),YELLOW,23),
                        text(f"raw[{index}]={desc.raw[index]:.5f}",
                             (2.0,-2.55),GREEN,22))
        self.hold(5)
        self.section("Từ 16 histogram đến vector 128 chiều",
                     "Trải phẳng theo thứ tự ô hàng, ô cột, rồi bin hướng")
        self.hero_to(patch,center=(-5.25,.25),width=3.0)
        vector=vector_map(desc.raw,center=(2.0,.25),cell=.43)
        self.add_detail(vector,text("16 ô × 8 bin = 128 số thực",(2.0,2.65),GREEN,25),
                        text(f"ô [{rr},{cc}], bin {bb} → vector[{index}]",
                             (2.0,-2.4),YELLOW,22))
        self.play(Indicate(vector[index],color=YELLOW),run_time=.65)
        self.hold(14)

    def normalization(self):
        self.section("Chuẩn hóa: L2 → cắt 0.2 → L2",
                     "Giảm độ nhạy với một phần thay đổi cường độ sáng")
        desc=self.data.descriptor
        arrays=(desc.raw,desc.first_l2,desc.clipped,desc.final)
        names=("raw","L2 lần 1","min(value, 0.2)","L2 lần 2")
        self.hero_to(self.data.image,center=(-5.3,.2),width=2.7)
        heat=vector_map(arrays[0],center=(1.5,.3),cell=.48)
        self.add_detail(heat)
        label=None
        for i,(name,values) in enumerate(zip(names,arrays)):
            if label is not None:
                self.play(FadeOut(label),run_time=.16)
            label=text(f"{name} · ||v||2 = {np.linalg.norm(values):.4f}",
                       (1.5,2.75),(ORANGE,CYAN,PURPLE,GREEN)[i],24)
            target=vector_map(values,center=(1.5,.3),cell=.48,
                              color=(ORANGE,CYAN,PURPLE,GREEN)[i])
            self.play(Transform(heat,target),FadeIn(label),run_time=.75)
            self.hold(3.5)
        self.details.add(label)
        idx=int(np.argmax(desc.first_l2))
        self.add_detail(text(f"v[{idx}]: {desc.raw[idx]:.3f} → "
                             f"{desc.first_l2[idx]:.3f} → {desc.clipped[idx]:.3f}"
                             f" → {desc.final[idx]:.3f}",
                             (1.5,-2.38),YELLOW,21))
        self.hold(7)

    def matching(self):
        self.section("So khớp với ảnh đã xoay và thu nhỏ",
                     "Phát hiện và mô tả độc lập trên A và B")
        self.step(8)
        d=self.data
        self.hero_to(d.image,center=(-4.65,.3),width=3.9)
        target=image_mobject(d.transformed,width=3.9).move_to([4.65,.3,0])
        self.add_detail(target,text(f"A · {len(d.described)} descriptor",
                                    (-4.65,2.65),CYAN,23),
                        text(f"B · {len(d.described_b)} descriptor",
                             (4.65,2.65),PURPLE,23))
        marks_a=VGroup(*[Dot(image_point(self.hero,d.image.shape,p.row_global,p.col_global),
                             radius=.045,color=CYAN) for p in d.described])
        marks_b=VGroup(*[Dot(image_point(target,d.transformed.shape,p.row_global,p.col_global),
                             radius=.045,color=PURPLE) for p in d.described_b])
        self.add_detail(marks_a,marks_b)
        self.hold(7)
        m=d.accepted; a=d.described[m.query]; b=d.described_b[m.nearest]
        qa=Dot(image_point(self.hero,d.image.shape,a.row_global,a.col_global),
               radius=.095,color=YELLOW)
        qb=Dot(image_point(target,d.transformed.shape,b.row_global,b.col_global),
               radius=.095,color=GREEN)
        second=d.described_b[m.second]
        q2=Dot(image_point(target,d.transformed.shape,second.row_global,second.col_global),
               radius=.085,color=ORANGE)
        distances=text(f"d1={m.d1:.4f} · d2={m.d2:.4f}",(0,-2.02),YELLOW,23)
        self.add_detail(qa,qb,q2,distances)
        self.hold(5)
        line=Line(qa.get_center(),qb.get_center(),color=GREEN,stroke_width=2.5)
        accept=text(f"d1/d2={m.ratio:.3f} < 0.8 → ACCEPT",(0,-2.65),GREEN,23)
        self.add_detail(line,accept)
        self.hold(6)
        expected=d.affine@np.array([a.col_global,a.row_global,1.])
        truth=Dot(image_point(target,d.transformed.shape,expected[1],expected[0]),
                  radius=.055,color=ORANGE)
        error_label=text(f"Kiểm tra hình học: sai số {d.geometry_error:.3f} px",
                         (0,3.25),ORANGE,22)
        self.add_detail(truth,error_label)
        self.hold(5)
        if d.rejected is not None:
            r=d.rejected
            reject_label=text(f"Một truy vấn khác: {r.ratio:.3f} ≥ 0.8 → REJECT",
                              (0,-1.44),MUTED,22)
            self.add_detail(reject_label)
            self.hold(3)
            self.play(FadeOut(reject_label),run_time=.2)
            self.remove(reject_label)
        actual=sift.match_descriptors(d.described,d.described_b)
        false=[(sift.evaluate_geometric_correspondence(d.described[z.query],
                d.described_b[z.nearest],d.affine)[0],z) for z in actual if z.accepted]
        false=[item for item in false if item[0]>4]
        if false:
            bad_error,bad=max(false,key=lambda item:item[0])
            bad_a=d.described[bad.query];bad_b=d.described_b[bad.nearest]
            bad_xy=d.affine@np.array([bad_a.col_global,bad_a.row_global,1.])
            bad_src=image_point(self.hero,d.image.shape,bad_a.row_global,bad_a.col_global)
            bad_dst=image_point(target,d.transformed.shape,bad_b.row_global,bad_b.col_global)
            bad_expected=image_point(target,d.transformed.shape,bad_xy[1],bad_xy[0])
            self.play(FadeOut(distances),FadeOut(accept),FadeOut(error_label),
                      FadeOut(q2),run_time=.2)
            self.play(qa.animate.move_to(bad_src),qb.animate.move_to(bad_dst),
                      truth.animate.move_to(bad_expected),
                      Transform(line,Line(bad_src,bad_dst,color=ORANGE,stroke_width=2.5)),
                      run_time=.85)
            self.add_detail(text(f"Ratio {bad.ratio:.3f} < 0.8: ACCEPT",
                                 (0,-2.02),GREEN,22),
                            text(f"Nhưng sai số hình học {bad_error:.2f} px: ghép sai",
                                 (0,-2.65),ORANGE,22))
            self.hold(6)

    def recap(self):
        self.section("SIFT: từ pixel đến so khớp",
                     "Vị trí, thang đo, hướng và descriptor cùng đến từ ảnh gốc")
        self.step(9)
        d=self.data
        self.hero_to(d.image,center=(-5.6,.15),width=3.5)
        sequence=(
            ("Ảnh gốc", d.image, CYAN),
            ("Gaussian · 18", d.pyramid[0].gaussians[2], PURPLE),
            ("DoG · 15", d.pyramid[0].dogs[1], ORANGE),
            ("Cực trị · 26", d.pyramid[0].dogs[d.candidate.layer], YELLOW),
        )
        for i,(name,values,color) in enumerate(sequence):
            x=-1.9+i*2.5
            thumb=image_mobject(values,width=1.8,signed=(i>=2)).move_to([x,.8,0])
            self.add_detail(thumb,text(name,(x,-.45),color,19))
            if i:
                arrow=Arrow([x-1.5,.8,0],[x-1.05,.8,0],buff=0,color=GRID)
                self.add_detail(arrow,run_time=.2)
            self.hold(1.1)
        refined=sum(p.status!="invalid" for p in d.filtered)
        self.add_detail(text(f"{len(d.raw)} thô → {refined} tinh chỉnh → "
                             f"{sum(p.status=='kept' for p in d.filtered)} giữ",
                             (2.15,-1.65),GREEN,21),
                        text(f"{len(d.described)} descriptor từ "
                             f"{sum(p.status=='kept' for p in d.filtered)} vị trí",
                             (2.15,-2.22),GREEN,20),
                        text("Hướng 36 bin → 4×4×8 → L2/clip/L2 → ratio",
                             (2.15,-2.75),WHITE,20))
        self.hold(11)
        self.section("Một vật thể, nhiều cách nhìn",
                     "SIFT mô tả đặc trưng cục bộ để phục vụ so khớp ảnh")
        self.hero_to(d.image,center=(-3.35,.2),width=4.3)
        self.add_detail(image_mobject(d.transformed,width=4.3).move_to([3.35,.2,0]),
                        text("Cùng máy ảnh · xoay 17° · tỷ lệ 0.94",
                             (0,-2.55),GREEN,23))
        self.hold(8)
