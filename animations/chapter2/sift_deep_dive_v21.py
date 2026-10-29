"""SIFT Deep Dive V2.1: computed visual derivations on the V2 camera fixture.

The numerical engines and the approved V2 scene are imported, never modified.
Run from this directory with the repository's root virtual environment.
"""

from __future__ import annotations

from collections import Counter
from math import cos, floor, hypot, radians, sin, sqrt

import cv2
import numpy as np
from manim import (Arrow, Circle, Create, Dot, FadeIn, FadeOut, Indicate, Line,
                   Rectangle, Rotate, Transform, Group as VGroup)

from chapter2_common import (CYAN, GREEN, GRID, MUTED, ORANGE, PURPLE,
                             WHITE, YELLOW, image_mobject, pixel_grid)
from sift_deep_dive import SIFTDeepDive, image_point, text, vector_map
from sift_deep_dive_data import build_data, gaussian_pixel_example
from utils import scale_space, sift


def fixed_gray(values):
    """Fixed [0,1] grayscale for comparable Gaussian intensities."""
    gray = np.round(np.clip(np.asarray(values), 0, 1) * 255).astype(np.uint8)
    return np.dstack((gray, gray, gray, np.full_like(gray, 255)))


def fixed_image(values, width):
    from manim import ImageMobject
    mob = ImageMobject(fixed_gray(values))
    mob.set_resampling_algorithm(0)
    mob.set(width=width)
    return mob


def histogram_bars(values, x0, y0, width, height, maximum, color):
    """Bars use one fixed numeric scale across partial/raw/smoothed stages."""
    a = np.asarray(values, dtype=float)
    return VGroup(*[
        Rectangle(width=width / len(a) * .76,
                  height=max(.006, float(v) / maximum * height),
                  stroke_width=0, fill_color=color, fill_opacity=.9)
        .move_to([x0 + (j + .5) * width / len(a),
                  y0 + max(.006, float(v) / maximum * height) / 2, 0])
        for j, v in enumerate(a)
    ])


def descriptor_heat(values, center, cell=.24, color=ORANGE):
    """Sixteen descriptor cells as rows, with eight orientation bins each."""
    a=np.asarray(values,dtype=float).reshape(16,8)
    peak=max(float(a.max()),1e-12)
    return VGroup(*[
        Rectangle(width=cell*.93,height=cell*.93,stroke_color=GRID,
                  stroke_width=.5,fill_color=color,
                  fill_opacity=.06+.86*float(a[r,c])/peak)
        .move_to([center[0]+(c-3.5)*cell,center[1]+(7.5-r)*cell,0])
        for r in range(16) for c in range(8)
    ])


def orientation_vote(sample):
    row, col, mag, angle, spatial, weight = sample
    u = angle / 10
    b0 = int(floor(u)) % 36
    fraction = u - floor(u)
    return b0, (b0 + 1) % 36, weight * (1 - fraction), weight * fraction


def descriptor_terms(sample):
    """The exact 2×2×2 weights from build_sift_descriptor()."""
    row, col, u, v, angle, weight = sample
    b = angle / 45
    result = []
    for rr in (floor(v), floor(v) + 1):
        for cc in (floor(u), floor(u) + 1):
            for bb in (floor(b), floor(b) + 1):
                if 0 <= rr < 4 and 0 <= cc < 4:
                    wy, wx, wb = 1 - abs(v - rr), 1 - abs(u - cc), 1 - abs(b - bb)
                    result.append((rr, cc, bb % 8, wy, wx, wb,
                                   weight * wy * wx * wb))
    return tuple(result)


def descriptor_sample_factors(sample, kp, octave):
    """Recover the engine's magnitude and spatial Gaussian for one saved sample."""
    row, col, _, _, _, weighted = sample
    image, _ = sift.gaussian_level(kp, octave)
    gx, gy = sift.gradient_field(image)
    magnitude = float(np.hypot(gx[row, col], gy[row, col]))
    width = 3 * kp.sigma_local
    dx, dy = col - kp.col_local, row - kp.row_local
    gaussian = float(np.exp(-(dx*dx + dy*dy) / (2*(2*width)**2)))
    assert np.isclose(magnitude * gaussian, weighted, atol=1e-12)
    return magnitude, gaussian, weighted


def trilinear_display(sample, term):
    """Nine decimal places keep even the smallest of eight votes nonzero."""
    _, _, _, wy, wx, wtheta, contribution = term
    factors = tuple(f"{value:.9f}" for value in (sample[5], wx, wy, wtheta))
    return factors, f"{contribution:.9f}"


class SIFTDeepDiveV21(SIFTDeepDive):
    """V2 story continuity with explicit numerical construction at hard stages."""

    def hero_to(self, values, center=(-4.7,.15), width=4.4, run_time=.75,
                signed=False):
        if signed:
            return super().hero_to(values,center,width,run_time,signed=True)
        display=np.asarray(values)
        if display.shape!=self.hero.pixel_array.shape[:2]:
            h,w=self.hero.pixel_array.shape[:2]
            display=cv2.resize(display,(w,h),interpolation=cv2.INTER_NEAREST)
        target=fixed_image(display,width).move_to([center[0],center[1],0])
        self.play(Transform(self.hero,target),run_time=run_time)

    def construct(self):
        self.data = build_data()
        self.storyboard = []
        self.heading = text("SIFT Deep Dive V2.1", (0, 3.91), WHITE, 33)
        self.caption = text("Từ pixel ảnh đến đặc trưng so khớp", (0, -3.42), WHITE, 26)
        names = ["Ảnh", "Gaussian", "Octave", "DoG", "Cực trị", "Tinh chỉnh",
                 "Hướng", "128D", "So khớp", "Tóm tắt"]
        self.progress_x = np.linspace(-6.6, 6.6, len(names))
        self.progress = VGroup(*[text(name, (float(x), -4.15), MUTED, 18)
                                 for x, name in zip(self.progress_x, names)])
        self.marker = Dot([self.progress_x[0], -4.43, 0], radius=.045, color=CYAN)
        self.hero = image_mobject(self.data.image, width=4.4).move_to([-4.7, .15, 0])
        self.details = VGroup()
        self.add(self.heading, self.caption, self.progress, self.marker, self.hero)
        self.opening()
        self.gaussian_pixel()
        self.increasing_sigma()
        self.gaussian_pyramid()
        self.downsample_detail()
        self.dog_pixel()
        self.dog_stacks()
        self.dog_usefulness()
        self.extrema()
        self.localization()
        self.filtering()
        self.orientation()
        self.rotating_frame()
        self.orientation_alignment()
        self.descriptor()
        self.normalization()
        self.matching()
        self.recap()

    def caption_to(self, message, color=WHITE):
        self.play(FadeOut(self.caption), run_time=.18)
        self.remove(self.caption)
        self.caption = text(message, (0, -3.42), color, 26)
        self.play(FadeIn(self.caption), run_time=.24)

    def remove_detail(self, *items):
        present = [item for item in items if item in self.mobjects]
        if present:
            self.play(*[FadeOut(item) for item in present], run_time=.20)
            self.remove(*present)
        for item in items:
            if item in self.details:
                self.details.remove(item)

    def gaussian_pixel(self):
        self.section("Gaussian: tính nhiều pixel liên tiếp",
                     "Một kernel quét ảnh, mỗi vị trí tạo đúng một pixel đầu ra")
        self.step(1)
        image = self.data.image
        blurred = self.data.pyramid[0].gaussians[0]
        kernel = scale_space.gaussian_kernel(1.6)
        output = fixed_image(blurred, 2.35).move_to([5.5, .55, 0])
        self.hero_to(image, center=(-5.5, .55), width=2.35)
        self.add_detail(text("Ảnh gốc", (-5.5, 2.15), CYAN, 22),
                        text("Kernel đầy đủ 15×15 · σ=1.6", (0, 2.15), PURPLE, 22),
                        text("Ảnh Gaussian [0,1]", (5.5, 2.15), GREEN, 22))
        crop_grid=pixel_grid(image[48:53,33:38],(-1.5,.55,0),
                             cell=.52,decimals=2,font_size=13)
        self.add_detail(crop_grid,
                        pixel_grid(kernel[6:9,6:9],(1.45,.55,0),
                                    cell=.60,decimals=3,font_size=14,accent=PURPLE),
                        text("Trích 5×5 từ ảnh",(-1.5,-1.04),CYAN,19),
                        text("Trích 3×3 từ kernel 15×15",(1.45,-1.04),PURPLE,19))
        self.add_detail(text("15 × 15 = 225 trọng số · đủ 225 pixel nguồn",
                             (0, -1.66), WHITE, 21))
        source_mark = Dot(radius=.075, color=YELLOW)
        source_mark.move_to(image_point(self.hero, image.shape, 50, 35))
        self.add_detail(source_mark)
        made=[]
        for n,(r, c) in enumerate(((50,35), (50,38), (65,65), (79,94))):
            patch, _, total = gaussian_pixel_example(self.data, r, c)
            assert np.isclose(total, blurred[r,c], atol=1e-10)
            new_crop=pixel_grid(image[r-2:r+3,c-2:c+3],(-1.5,.55,0),
                                cell=.52,decimals=2,font_size=13)
            self.play(source_mark.animate.move_to(image_point(self.hero,image.shape,r,c)),
                      Transform(crop_grid,new_crop),run_time=.65)
            note = self.add_detail(text(f"[{r},{c}]: {patch[7,7]:.3f} → Σ(I·K) = {total:.4f}",
                                        (0,-2.38),YELLOW,22),run_time=.25)
            made.append(self.add_detail(text(f"L[{r},{c}] = {total:.4f}",
                                             (5.5,1.44-n*.56),GREEN,21),run_time=.25))
            self.hold(2.2)
            self.remove_detail(note)
        self.remove_detail(*made)
        self.add_detail(output)
        self.caption_to("Cùng kernel, kết quả thay đổi theo lân cận pixel")
        self.hold(5)
        self.hero_to(blurred, center=(-5.5,.55), width=2.35)
        self.hold(3)

    def downsample_detail(self):
        self.section("Octave mới = lấy mẫu ::2 thật",
                     "Giữ pixel hàng chẵn, cột chẵn của G(0,3)")
        parent=self.data.pyramid[0].gaussians[3]
        child=self.data.pyramid[1].gaussians[0]
        np.testing.assert_array_equal(child,parent[::2,::2])
        self.hero_to(parent,center=(-5.5,.55),width=2.55)
        result=fixed_image(child,2.55).move_to([5.5,.55,0])
        self.add_detail(result,text("G(0,3) · 129²",(-5.5,2.20),PURPLE,21),
                        text("G(1,0) · 65²",(5.5,2.20),GREEN,21))
        patch=parent[46:52,46:52]
        grid=pixel_grid(patch,(0,.65,0),cell=.43,decimals=2,font_size=13)
        self.add_detail(grid)
        self.hold(2.5)
        selected=[];dim=[]
        for r in range(6):
            for c in range(6):
                if r%2==0 and c%2==0:
                    selected.append(grid[r*6+c])
                else:
                    dim.append(grid[r*6+c])
        self.play(*[cell[0].animate.set_fill(opacity=.10) for cell in dim],
                  *[cell[1].animate.set_opacity(.15) for cell in dim],
                  run_time=.7)
        self.caption_to("Chín pixel còn sáng: [46,48,50] × [46,48,50]")
        for cell in selected:
            self.play(Indicate(cell,color=YELLOW),run_time=.22)
        self.hold(2.5)
        sample=parent[46:52:2,46:52:2]
        output_grid=pixel_grid(sample,(5.5,-1.91,0),cell=.40,
                               decimals=2,font_size=13)
        self.add_detail(output_grid,
                        text("Cùng số liệu ở mảng 65²",(5.5,-2.77),GREEN,18))
        self.caption_to("Độ phân giải giảm thật; ảnh bên phải phóng lớn để dễ xem")
        self.hold(4)

    def dog_usefulness(self):
        self.section("Một pixel qua năm tầng DoG",
                     "Cùng tọa độ [row,col] được đo ở năm thang làm mờ")
        d = self.data; kp = d.candidate; octave = d.pyramid[kp.octave]
        values = np.array([a[kp.row,kp.col] for a in octave.dogs])
        self.dog_values = values.copy()
        self.play(self.hero.animate.set_opacity(0),run_time=.2)
        tiles = []
        marks = []
        roots = []
        for j, array in enumerate(octave.dogs):
            x = -5.55 + j*1.25
            tile = image_mobject(array, width=1.02, signed=True).move_to([x,1.25,0])
            mark = Circle(radius=.07, color=YELLOW, stroke_width=2).move_to(
                image_point(tile,array.shape,kp.row,kp.col))
            tiles.append(tile); marks.append(mark)
            roots.append(self.add_detail(tile,mark,text(f"D{j}",(x,2.16),PURPLE,20),run_time=.17))
        self.dog_tiles = tiles
        self.dog_marks = marks
        self.dog_tile_roots = roots
        self.play(*[root.animate.set_opacity(.34) for root in roots],run_time=.3)
        self.add_detail(text(f"Cùng pixel [{kp.row},{kp.col}]",(-3.1,2.82),YELLOW,21))
        x0,dx=.75,1.27; zero=1.75; factor=41
        self.add_detail(Line([.45,zero,0],[6.65,zero,0],color=GRID),
                        text("0",(.15,zero),MUTED,17),
                        text("DoG có dấu",(3.6,2.65),WHITE,21),
                        text("Tầng DoG →",(3.6,-2.65),WHITE,20))
        for v in (-.04,-.08):
            y=zero+v*factor
            self.add_detail(Line([.45,y,0],[6.65,y,0],color=GRID,stroke_width=1),
                            text(f"{v:+.2f}",(.05,y),MUTED,16),run_time=.1)
        dots=[]
        for j,value in enumerate(values):
            x=x0+j*dx; y=zero+value*factor
            active=[roots[j].animate.set_opacity(1)]
            if j:
                active.append(roots[j-1].animate.set_opacity(.34))
            self.play(*active,run_time=.20)
            crop=octave.dogs[j][kp.row-2:kp.row+3,kp.col-2:kp.col+3]
            zoom=image_mobject(crop,width=1.15,signed=True).move_to([-3.15,-1.05,0])
            zoom_mark=Circle(radius=.11,color=YELLOW).move_to(zoom)
            detail=self.add_detail(zoom,zoom_mark,
                text(f"D{j}[{kp.row},{kp.col}] = {value:+.5f}",
                     (-3.15,-2.18),CYAN if value<0 else ORANGE,19),run_time=.2)
            dot=Dot([x,y,0],radius=.075,color=CYAN if value<0 else ORANGE)
            dot_root=self.add_detail(dot,text(f"D{j}",(x,-2.27),PURPLE,18),run_time=.28)
            dots.append(dot)
            if j==1:
                self.dog_selected_point_root=dot_root
            if len(dots)>1:
                self.add_detail(Line(dots[-2].get_center(),dot.get_center(),
                                     color=YELLOW,stroke_width=2),run_time=.18)
            self.hold(2.35)
            self.remove_detail(detail)
        self.dog_graph_dots = dots
        ring=Circle(radius=.17,color=YELLOW).move_to(dots[1])
        self.add_detail(ring,text(f"D1 = {values[1]:+.5f}",(3.7,2.12),YELLOW,22))
        self.caption_to("D1 thấp hơn D0, D2 tại pixel này; chưa đủ kết luận cực trị 3D")
        self.hold(8)

    def extrema(self):
        """Carry the five measured DoG layers into their exact 3×3 neighborhoods."""
        self.storyboard.append(("Cực trị 3D · 26 láng giềng",float(self.renderer.time)))
        self.play(FadeOut(self.heading),FadeOut(self.caption),run_time=.18)
        self.remove(self.heading,self.caption)
        self.heading=text("Cực trị 3D · 26 láng giềng",(0,3.91),WHITE,33)
        self.caption=text("Giữ D0, D1, D2 tại đúng pixel vừa đo",(0,-3.42),WHITE,26)
        self.play(FadeIn(self.heading),FadeIn(self.caption),run_time=.25)
        self.step(4)
        d=self.data; kp=d.candidate; octave=d.pyramid[kp.octave]
        # Retain the D1 chart point until the three source patches appear.
        keep=set(self.dog_tile_roots[:3]+[self.dog_selected_point_root])
        extras=[m for m in self.mobjects if m not in keep and
                m not in (self.heading,self.caption,self.progress,self.marker,self.hero)]
        self.remove_detail(*extras)
        self.play(*[self.dog_tile_roots[j].animate.shift(
                    np.array([-3.9+j*3.9,2.1,0])-self.dog_tiles[j].get_center())
                    .set_opacity(1) for j in range(3)],run_time=.7)
        regions=[]
        for j,x in enumerate((-3.9,0,3.9)):
            grid=pixel_grid(octave.dogs[j][kp.row-1:kp.row+2,kp.col-1:kp.col+2]*1000,
                            (x,.15,0),cell=.67,decimals=1,signed=True,font_size=16)
            regions.append(grid)
            self.add_detail(Line(self.dog_tiles[j].get_bottom(),grid.get_top(),
                                 color=YELLOW,stroke_width=1.5),grid,
                            text(f"D{j} × 1000",(x,1.23),PURPLE,20),run_time=.5)
        self.play(self.dog_graph_dots[1].animate.move_to(regions[1][4]),
                  run_time=.85)
        self.hold(3)
        self.remove_detail(self.dog_selected_point_root)
        center=float(octave.dogs[kp.layer][kp.row,kp.col]*1000)
        neighbors=scale_space.inspect_neighbors(octave.dogs,kp.layer,kp.row,kp.col)
        lowest=min(v.value*1000 for v in neighbors)
        self.add_detail(Circle(radius=.27,color=YELLOW).move_to(regions[1][4]),
                        text(f"p={center:+.2f} · min láng giềng={lowest:+.2f}",
                             (0,-1.32),YELLOW,22))
        groups=(VGroup(*regions[1][:4],*regions[1][5:]),regions[0],regions[2])
        count=0
        for group,increment,label in zip(groups,(8,9,9),
                                          ("8 cùng tầng D1","9 ở D0","9 ở D2")):
            count+=increment
            status=self.add_detail(text(f"{label} · compared {count}/26",
                                        (0,-2.25),CYAN,22),run_time=.15)
            self.play(group.animate.set_opacity(.7),run_time=.35)
            self.hold(4.2)
            self.remove_detail(status)
        assert kp.kind=="MIN" and center<lowest
        self.caption_to("p nhỏ hơn cả 26 điểm: ứng viên cực tiểu DoG, chưa là keypoint cuối")
        self.hold(7)

    def filtering(self):
        self.section("Lọc ứng viên: yếu · biên · ổn định",
                     "Ba quyết định dùng phản hồi thật từ ảnh máy ảnh A và B")
        d=self.data
        low=next(p for p in d.filtered_b if p.status=="low contrast")
        edge=next(p for p in d.filtered if p.status=="edge")
        good=d.selected
        cases=((low,d.pyramid_b,"B · yếu",MUTED),
               (edge,d.pyramid,"A · giống biên",ORANGE),
               (good,d.pyramid,"A · ổn định",GREEN))
        self.hero_to(d.image,center=(-6.8,2.15),width=.9)
        for x,(kp,pyramid,title,color) in zip((-4.3,0,4.3),cases):
            array=pyramid[kp.octave].dogs[kp.layer]
            r,c=kp.row,kp.col
            crop=array[max(0,r-10):r+11,max(0,c-10):c+11]
            if crop.size==0: continue
            image=image_mobject(crop,width=2.45,signed=True).move_to([x,.7,0])
            self.add_detail(image,text(title,(x,2.25),color,23),
                            text(f"[{r},{c}] · {kp.status}",(x,-.90),color,19))
            self.hold(1.3)
        h=sift.calculate_spatial_hessian(d.pyramid[edge.octave].dogs,
                                          edge.layer,edge.row,edge.col)
        decision=sift.edge_decision(h)
        assert abs(low.response)<.008 and edge.status=="edge" and good.status=="kept"
        self.add_detail(text(f"|D_interp|={abs(low.response):.5f} < 0.008",
                             (-4.3,-1.56),MUTED,20),
                        text(f"2×2: trace²/det={decision[3]:.2f}",
                             (0,-1.56),ORANGE,20),
                        text(f"|D_interp|={abs(good.response):.5f}",
                             (4.3,-1.56),GREEN,20))
        self.caption_to("3×3 Hessian định vị; 2×2 Hessian kiểm tra dạng biên")
        self.hold(5)
        count=Counter(p.status for p in d.filtered)
        refined=len(d.raw)-count['invalid']
        self.add_detail(text(f"A: {len(d.raw)} thô → {refined} tinh chỉnh → "
                             f"{count['kept']} vị trí → {len(d.described)} descriptor",
                             (0,-2.43),YELLOW,21))
        self.hold(4)

    def localization(self):
        super().localization()
        kp=self.data.selected; raw=self.data.candidate
        assert (raw.col,kp.col)==(103,102)
        self.caption_to("Lặp lại quanh cột 102 rồi mới áp δ đo tại tâm mới")
        self.add_detail(text(f"x={kp.col}+{kp.delta[0]:.3f}={kp.col_local:.3f}; "
                             f"y={kp.row}+{kp.delta[1]:.3f}={kp.row_local:.3f}",
                             (2.3,-.04),GREEN,20))
        self.hold(5)

    def orientation(self):
        d=self.data; kp=d.selected; info=d.orientation
        self.section("Hướng: histogram hình thành từ gradient",
                     "Tính từ pixel Gaussian thật rồi bỏ phiếu vào 36 bin")
        self.step(6)
        image,j=sift.gaussian_level(kp,d.pyramid[kp.octave])
        gx,gy=sift.gradient_field(image)
        self.hero_to(image,center=(-6.1,1.45),width=1.65)
        self.add_detail(text(f"G({kp.octave},{j}) · keypoint [{kp.row},{kp.col}]",
                             (-4.85,2.70),CYAN,19),
                        Circle(radius=.06,color=YELLOW).move_to(
                            image_point(self.hero,image.shape,kp.row_local,kp.col_local)))
        r0,c0=kp.row-5,kp.col-5
        crop=image[r0:r0+11,c0:c0+11]
        zoom=fixed_image(crop,2.55).move_to([-4.55,-.55,0])
        self.add_detail(zoom,text("Phóng to 11×11 pixel",(-4.55,-2.15),WHITE,19))
        top=sorted(info.contributions,key=lambda s:-s[5])[:3]
        maximum=max(float(info.raw_histogram.max()),float(info.histogram.max()))*1.04
        x0,y0,width,height=.05,-1.60,6.65,2.35
        self.add_detail(Line([x0,y0,0],[x0+width,y0,0],color=GRID),
                        text("0°",(x0,-1.95),MUTED,16),
                        text("90°",(x0+width*9/36,-1.95),MUTED,16),
                        text("180°",(x0+width*18/36,-1.95),MUTED,16),
                        text("270°",(x0+width*27/36,-1.95),MUTED,16))
        hist_title=self.add_detail(text("36 bin × 10°",(3.35,2.65),PURPLE,22))
        cumulative=np.zeros(36)
        hist=histogram_bars(cumulative,x0,y0,width,height,maximum,PURPLE)
        self.add_detail(hist)
        field=VGroup()
        for sample in top:
            row,col,mag,angle,spatial,weight=sample
            start=image_point(zoom,crop.shape,row-r0,col-c0)
            vec=np.array([gx[row,col],-gy[row,col],0])
            length=mag*2.7
            endpoint=start+vec/(mag or 1)*length
            arrow=Arrow(start,endpoint,buff=0,color=YELLOW,stroke_width=3,
                        max_tip_length_to_length_ratio=.25)
            field.add(arrow)
            self.add_detail(arrow,Dot(start,radius=.047,color=YELLOW),run_time=.3)
            ix=(image[row,col+1]-image[row,col-1])/2
            iy=(image[row+1,col]-image[row-1,col])/2
            assert np.isclose(ix,gx[row,col]) and np.isclose(iy,gy[row,col])
            calc=self.add_detail(
                text(f"[{row},{col}] Ix=({image[row,col+1]:.3f}−"
                     f"{image[row,col-1]:.3f})/2 = {ix:+.3f}",
                     (3.35,2.30),YELLOW,18),
                text(f"Iy=({image[row+1,col]:.3f}−"
                     f"{image[row-1,col]:.3f})/2 = {iy:+.3f}",
                     (3.35,1.81),YELLOW,18),
                text(f"m={mag:.3f} · atan2(Iy,Ix)={angle:.1f}°",
                     (3.35,1.30),GREEN,18),run_time=.2)
            self.hold(3.0)
            self.remove_detail(calc)
        self.caption_to("Mũi tên: (Ix, −Iy) trên màn hình vì ảnh có trục y hướng xuống")
        more=sorted(info.contributions,key=lambda s:-s[5])[3:9]
        for group in (more[:3],more[3:]):
            new=[]
            for row,col,mag,angle,spatial,weight in group:
                start=image_point(zoom,crop.shape,row-r0,col-c0)
                vector=np.array([gx[row,col],-gy[row,col],0])
                new.append(Arrow(start,start+vector/(mag or 1)*mag*2.7,
                                 buff=0,color=CYAN,stroke_width=2.5))
            self.add_detail(*new,run_time=.6)
            self.hold(1.3)
        one=top[0]
        dist=hypot(one[1]-kp.col_local,one[0]-kp.row_local)
        weighted=self.add_detail(
            text(f"d={dist:.2f}px; σ_w=1.5σ={1.5*kp.sigma_local:.2f}",
                 (3.35,2.20),PURPLE,19),
            text(f"w=exp(−d²/2σ_w²)={one[4]:.3f}",
                 (3.35,1.70),PURPLE,19),
            text(f"vote=m×w={one[2]:.3f}×{one[4]:.3f}={one[5]:.3f}",
                 (3.35,1.20),YELLOW,19))
        self.caption_to("Gaussian không gian giảm ảnh hưởng của mẫu xa keypoint")
        self.hold(4)
        self.remove_detail(weighted)
        for n,sample in enumerate(top):
            row,col,mag,angle,spatial,weight=sample
            b0,b1,v0,v1=orientation_vote(sample)
            note=self.add_detail(text(f"[{row},{col}]: {mag:.3f}×{spatial:.3f}={weight:.3f}; "
                                      f"{angle:.1f}° → bin {b0},{b1}",
                                      (3.35,1.92),YELLOW,18),
                                 text(f"{v0:.3f} + {v1:.3f} = {weight:.3f}",
                                      (3.35,1.43),GREEN,18),run_time=.18)
            cumulative[b0]+=v0;cumulative[b1]+=v1
            target=histogram_bars(cumulative,x0,y0,width,height,maximum,PURPLE)
            self.play(Transform(hist,target),run_time=.65)
            self.hold(2.3)
            self.remove_detail(note)
        self.caption_to("Các gradient còn lại tiếp tục cộng vào đúng hai bin kề nhau")
        ordered=sorted(info.contributions,key=lambda s:-s[5])
        for limit in (12,50,150):
            cumulative=np.zeros(36)
            for sample in ordered[:limit]:
                b0,b1,v0,v1=orientation_vote(sample)
                cumulative[b0]+=v0;cumulative[b1]+=v1
            status=self.add_detail(text(f"Đã cộng {limit}/{len(ordered)} gradient",
                                        (3.35,1.92),CYAN,20),run_time=.17)
            self.play(Transform(hist,histogram_bars(cumulative,x0,y0,width,height,
                                                   maximum,PURPLE)),run_time=.70)
            self.hold(1.1)
            self.remove_detail(status)
        self.play(Transform(hist,histogram_bars(info.raw_histogram,x0,y0,width,height,
                                               maximum,PURPLE)),run_time=1.15)
        raw_count=self.add_detail(text(f"{len(info.contributions)} mẫu → histogram thô",
                                       (3.35,1.92),WHITE,21))
        self.hold(4)
        self.caption_to("Làm mượt vòng tròn bằng trọng số 1 : 4 : 6 : 4 : 1")
        self.remove_detail(raw_count)
        peak=int(np.argmax(info.histogram))
        raw=info.raw_histogram
        smooth_calc=self.add_detail(
            text(f"bin {peak}: [1,4,6,4,1] · raw[{peak-2}:{peak+3}]/16",
                 (3.35,2.05),WHITE,18),
            text(f"{raw[peak-2]:.3f}, {raw[peak-1]:.3f}, {raw[peak]:.3f}, "
                 f"{raw[(peak+1)%36]:.3f}, {raw[(peak+2)%36]:.3f}"
                 f" → {info.histogram[peak]:.3f}",
                 (3.35,1.55),YELLOW,18))
        self.play(Transform(hist,histogram_bars(info.histogram,x0,y0,width,height,
                                               maximum,PURPLE)),run_time=1.1)
        self.hold(4)
        self.remove_detail(smooth_calc)
        assert peak*10 in info.peaks
        self.remove_detail(hist_title,raw_count)
        peak_x=x0+(peak+.5)*width/36
        arrow=Arrow([peak_x,2.1,0],hist[peak].get_top()+[0,.04,0],
                    buff=0,color=YELLOW,stroke_width=3)
        self.add_detail(arrow,text(f"bin {peak} = {10*peak}° · đỉnh {info.histogram[peak]:.3f}",
                                   (3.4,2.57),YELLOW,20))
        self.play(Indicate(hist[peak],color=YELLOW),run_time=.65)
        self.caption_to("Đỉnh phụ phải là cực đại cục bộ và đạt ≥80% đỉnh lớn nhất")
        self.hold(4)
        alternative=next(p for p in d.filtered if p.status=="kept" and
                         p.octave==kp.octave and p is not kp and
                         len(sift.calculate_orientation_histogram(
                             p,d.pyramid[p.octave]).peaks)==2 and
                         sift.gaussian_level(p,d.pyramid[p.octave])[1]==j)
        alt=sift.calculate_orientation_histogram(alternative,d.pyramid[alternative.octave])
        self.section("Một vị trí có thể sinh hai bản ghi hướng",
                     "Ví dụ khác: cùng một keypoint, hai đỉnh hợp lệ")
        self.hero_to(image,center=(-4.7,.25),width=3.5)
        anchor=image_point(self.hero,image.shape,alternative.row_local,
                           alternative.col_local)
        self.add_detail(Dot(anchor,radius=.10,color=GREEN),
                        text(f"[{alternative.row},{alternative.col}]",
                             (-4.7,2.4),YELLOW,21))
        for i,angle in enumerate(alt.peaks):
            vec=np.array([cos(radians(angle)),-sin(radians(angle)),0])
            self.add_detail(Arrow(anchor,anchor+.72*vec,buff=0,
                                  color=(YELLOW,PURPLE)[i]),
                            text(f"{angle:.0f}°",(1.25+i*2.2,2.3),
                                 (YELLOW,PURPLE)[i],23),run_time=.45)
        self.add_detail(histogram_bars(alt.histogram,-.3,-1.4,6.5,2.2,
                                       float(max(alt.raw_histogram.max(),alt.histogram.max()))*1.04,
                                       PURPLE))
        self.hold(5)

    def rotating_frame(self):
        self.section("Xoay hệ tọa độ descriptor quanh keypoint",
                     "Ảnh Gaussian đứng yên; khung lấy mẫu xoay theo hướng đã chọn")
        d=self.data; kp=d.oriented
        image,j=sift.gaussian_level(kp,d.pyramid[kp.octave])
        self.hero_to(image,center=(-4.7,.25),width=3.9)
        anchor=image_point(self.hero,image.shape,kp.row_local,kp.col_local)
        span=4*3*kp.sigma_local*self.hero.width/image.shape[1]
        square=Rectangle(width=span,height=span,stroke_color=PURPLE,
                         stroke_width=2).move_to(anchor)
        xaxis=Arrow(anchor,anchor+[span/2,0,0],buff=0,color=YELLOW)
        yaxis=Arrow(anchor,anchor+[0,-span/2,0],buff=0,color=CYAN)
        frame=VGroup(square,xaxis,yaxis)
        self.add_detail(Dot(anchor,radius=.09,color=YELLOW),frame,
                        text(f"θ={kp.orientation:.0f}° · ô rộng 3σ={3*kp.sigma_local:.2f}px",
                             (2.45,2.65),PURPLE,21))
        self.hold(2)
        self.play(Rotate(frame,angle=-radians(kp.orientation),about_point=anchor),
                  run_time=1.1)
        patch,_=sift.sample_oriented_descriptor_patch(kp,d.pyramid[kp.octave],48)
        self.add_detail(fixed_image(patch,3.15).move_to([2.6,.2,0]),
                        text("Patch lấy mẫu theo khung đã xoay",(2.6,-1.8),GREEN,20))
        self.hold(5)

    def orientation_alignment(self):
        """A measured A/B pair shows why local frames are rotated."""
        d=self.data; m=d.accepted
        a=d.described[m.query]; b=d.described_b[m.nearest]
        self.section("Hai ảnh, hai khung hướng cục bộ",
                     "Ví dụ so khớp khác: mỗi ảnh tự gán hướng cho keypoint")
        self.hero_to(d.image,center=(-5.75,.35),width=2.2)
        image_b=image_mobject(d.transformed,width=2.2).move_to([5.75,.35,0])
        self.add_detail(image_b,
                        Dot(image_point(self.hero,d.image.shape,a.row_global,
                                        a.col_global),radius=.075,color=YELLOW),
                        Dot(image_point(image_b,d.transformed.shape,b.row_global,
                                        b.col_global),radius=.075,color=YELLOW),
                        text(f"A: {a.orientation:.0f}°",(-5.75,2.16),PURPLE,20),
                        text(f"B: {b.orientation:.0f}°",(5.75,2.16),PURPLE,20))
        self.hold(3)
        pa,_=sift.sample_oriented_descriptor_patch(a,d.pyramid[a.octave],48)
        pb,_=sift.sample_oriented_descriptor_patch(b,d.pyramid_b[b.octave],48)
        left=fixed_image(pa,2.35).move_to([-1.55,.25,0])
        right=fixed_image(pb,2.35).move_to([1.55,.25,0])
        self.add_detail(left,right,
                        text("Khung đã căn hướng A",(-1.55,2.05),CYAN,18),
                        text("Khung đã căn hướng B",(1.55,2.05),GREEN,18))
        self.caption_to("Xoay hệ tọa độ cục bộ giúp so cấu trúc khi vật thể xoay")
        self.hold(6)
        self.add_detail(text("Mỗi patch được nội suy từ Gaussian tương ứng",
                             (0,-2.62),YELLOW,20))
        self.hold(4)

    def descriptor(self):
        d=self.data; kp=d.oriented; desc=d.descriptor
        self.section("Descriptor: 16 ô, mỗi ô 8 hướng",
                     "Mỗi gradient đóng góp vào các ô và bin lân cận")
        self.step(7)
        patch,_=sift.sample_oriented_descriptor_patch(kp,d.pyramid[kp.octave],48)
        self.hero_to(patch,center=(-4.75,.3),width=3.55)
        cells=VGroup(*[
            Rectangle(width=.8875,height=.8875,stroke_color=PURPLE,
                      stroke_width=1.5).move_to([-4.75+(c-1.5)*.8875,
                                                 .3+(1.5-r)*.8875,0])
            for r in range(4) for c in range(4)
        ])
        self.add_detail(cells,text("16 ô · 8 bin/ô · 45°/bin",(2.15,2.6),PURPLE,23),
                        text(f"Ô rộng 3σ = {3*kp.sigma_local:.2f} pixel nguồn",
                             (2.15,1.93),WHITE,21))
        self.hold(3)
        rr,cc,bb=np.unravel_index(int(np.argmax(desc.cells)),desc.cells.shape)
        index=sift.descriptor_index(int(rr),int(cc),int(bb))
        contributions=[]
        for sample in desc.samples:
            for term in descriptor_terms(sample):
                if term[:3]==(rr,cc,bb):
                    contributions.append((term[-1],sample,term))
        contributions.sort(key=lambda item:-item[0])
        assert np.isclose(sum(item[0] for item in contributions),desc.raw[index])
        self.descriptor_example_index=index
        self.descriptor_example_terms=tuple(item[0] for item in contributions)
        sample=contributions[0][1]
        row,col,u,v,angle,weight=sample
        gx,gy=sift.gradient_field(sift.gaussian_level(kp,d.pyramid[kp.octave])[0])
        mag=float(np.hypot(gx[row,col],gy[row,col]))
        absolute=float(np.degrees(np.arctan2(gy[row,col],gx[row,col]))%360)
        assert np.isclose((absolute-kp.orientation)%360,angle)
        sample_point=image_point(self.hero,patch.shape,(v+.5)*12,(u+.5)*12)
        active=cells[rr*4+cc].copy().set_stroke(YELLOW,3)
        self.add_detail(Dot(sample_point,radius=.09,color=YELLOW),active,
                        text(f"Mẫu [{row},{col}] · Ix={gx[row,col]:+.3f}, Iy={gy[row,col]:+.3f}",
                             (2.15,1.19),YELLOW,18),
                        text(f"m={mag:.3f}; θ_abs={absolute:.1f}°; "
                             f"θ_rel={angle:.1f}°",(2.15,.64),WHITE,18),
                        text(f"u={u:.3f}, v={v:.3f} trong khung xoay",
                             (2.15,.08),CYAN,20))
        self.caption_to("Góc descriptor = góc gradient trừ hướng keypoint")
        self.hold(4)

        # First show one real weighted contribution in all valid 2×2×2 bins.
        self.section("Nội suy ba chiều: hàng · cột · hướng",
                     "Một mẫu có thể chia phiếu cho tối đa 8 thành phần")
        self.hero_to(patch,center=(-5.25,.15),width=3.1)
        terms=descriptor_terms(sample)
        support=VGroup(*[
            Rectangle(width=.775,height=.775,stroke_color=GRID,
                      stroke_width=1).move_to([-5.25+(c-1.5)*.775,
                                              .15+(1.5-r)*.775,0])
            for r in range(4) for c in range(4)])
        self.add_detail(support)
        for row_index in sorted(set(t[0] for t in terms)):
            for col_index in sorted(set(t[1] for t in terms)):
                self.add_detail(support[row_index*4+col_index].copy()
                                .set_stroke(PURPLE,2),run_time=.24)
        magnitude, gaussian, weighted = descriptor_sample_factors(
            sample, kp, d.pyramid[kp.octave])
        sample_dot=Dot(image_point(self.hero,patch.shape,(v+.5)*12,(u+.5)*12),
                       radius=.105,color=YELLOW)
        self.add_detail(sample_dot,
                        text(f"Mẫu [{row},{col}] · m={magnitude:.9f}",
                             (2.55,2.63),YELLOW,20),
                        text(f"Gaussian={gaussian:.9f} → weighted={weighted:.9f}",
                             (2.55,2.12),CYAN,19),
                        text("weighted = m × Gaussian",(2.55,1.67),WHITE,19),
                        text("Đóng góp = weighted × wx × wy × wθ",
                             (2.55,1.21),WHITE,19))
        self.hold(1.1)
        # Same original gradient, two destinations with sharply different wy.
        for example in (terms[0], terms[4]):
            r,c,b,wy,wx,wb,part=example
            factors,result=trilinear_display(sample,example)
            target=support[r*4+c].copy().set_stroke(YELLOW,4)
            explanation=self.add_detail(
                target,
                text(f"Ô ({r},{c}) · bin {b}",(2.55,.68),YELLOW,24),
                text(f"wx={factors[1]} · wy={factors[2]}",
                     (2.55,.21),ORANGE,19),
                text(f"wθ={factors[3]}",(2.55,-.25),ORANGE,19),
                text(f"{factors[0]} × {factors[1]}",
                     (2.55,-.77),WHITE,20),
                text(f"× {factors[2]} × {factors[3]}",
                     (2.55,-1.24),WHITE,20),
                text(f"≈ {result}",(2.55,-1.79),GREEN,25),
                run_time=.35)
            self.hold(2.6)
            self.remove_detail(explanation)
        self.add_detail(text(f"Cùng mẫu [{row},{col}] → 8 ô/bin lân cận",
                             (2.55,.78),YELLOW,21))
        for k,term in enumerate(terms):
            r,c,b=term[:3]
            _,result=trilinear_display(sample,term)
            x=.92 if k<4 else 4.5
            y=.23-(k%4)*.57
            self.add_detail(text(f"({r},{c}), bin {b} → {result}",
                                 (x,y),ORANGE,19),run_time=.16)
        self.add_detail(text(f"Σ 8 phần ≈ {sum(t[-1] for t in terms):.9f} = weighted",
                             (2.55,-2.43),GREEN,21))
        self.hold(4)

        self.section("Histogram một ô: cộng nhiều mẫu",
                     "Một phần đóng góp khác với giá trị tích lũy cuối")
        self.hero_to(patch,center=(-5.25,.25),width=3.1)
        x0,y0,width,height=-.2,-1.40,6.5,2.3
        maximum=max(float(desc.cells[rr,cc].max())*1.06,.01)
        partial=np.zeros(8)
        hist=histogram_bars(partial,x0,y0,width,height,maximum,ORANGE)
        self.add_detail(hist,text(f"Ô ({rr},{cc}) · 8 bin × 45°",(3.0,2.60),PURPLE,23),
                        text(f"bin {bb} → vector[{index}]",(3.0,1.96),YELLOW,20))
        total=0
        for k,(part,one_sample,term) in enumerate(contributions[:8]):
            total+=part
            partial[bb]=total
            note=self.add_detail(text(f"Mẫu {k+1} [{one_sample[0]},{one_sample[1]}] "
                                      f"+{part:.5f} → {total:.5f}",
                                      (3.0,-2.23),YELLOW,20),run_time=.18)
            self.play(Transform(hist,histogram_bars(partial,x0,y0,width,height,
                                                   maximum,ORANGE)),run_time=.55)
            self.hold(1.7)
            self.remove_detail(note)
        self.play(Transform(hist,histogram_bars(desc.cells[rr,cc],x0,y0,width,
                                               height,maximum,ORANGE)),run_time=.9)
        self.add_detail(text(f"Σ {len(contributions)} phần → raw[{index}] = "
                             f"{desc.raw[index]:.5f}",(3.0,-2.23),GREEN,21))
        self.hold(6)

        self.section("16 histogram → vector 128 chiều",
                     "Trải phẳng: hàng ô, cột ô, rồi bin hướng")
        self.hero_to(patch,center=(-6.05,.25),width=1.65)
        mini=VGroup()
        for r in range(4):
            row_group=VGroup()
            for c in range(4):
                x=-2.9+c*1.25; y=1.95-r*1.3
                cell=desc.cells[r,c]
                local_max=max(float(desc.cells.max()),1e-12)
                row_group.add(histogram_bars(cell,x-.48,y-.38,.96,.75,local_max,
                                             ORANGE),
                              text(f"({r},{c})",(x,y+.47),PURPLE,15))
            mini.add(row_group)
        self.add_detail(text("4 × 4 ô, mỗi ô 8 cột",(-1.05,2.91),WHITE,20))
        for r,row_group in enumerate(mini):
            self.add_detail(row_group,run_time=.65)
            self.hold(2.0)
        heat=descriptor_heat(desc.raw,center=(4.60,.18),cell=.24)
        self.add_detail(heat,text("16 hàng × 8 bin = 128",(4.6,2.67),GREEN,21),
                        text("(0,0) → 0–7; (0,1) → 8–15",(4.6,-2.31),WHITE,18),
                        text("(3,3) → 120–127",(4.6,-2.77),WHITE,18))
        for idx in (0,8,index,127):
            mark=heat[idx]
            self.play(Indicate(mark,color=YELLOW),run_time=.45)
        self.hold(7)
        self.caption_to("Hướng keypoint: 1×36 bin; descriptor: 16×8 bin")
        self.hold(3)

    def normalization(self):
        self.section("Chuẩn hóa cùng một descriptor",
                     "L2 → giới hạn 0.2 → L2 lần nữa")
        desc=self.data.descriptor
        self.hero_to(self.data.image,center=(-7.0,.1),width=.85)
        arrays=(desc.raw,desc.first_l2,desc.clipped,desc.final)
        names=("raw","L2 lần 1","clip 0.2","L2 lần 2")
        colors=(ORANGE,CYAN,PURPLE,GREEN)
        heats=[]
        for i,(name,values,color) in enumerate(zip(names,arrays,colors)):
            x=(-3.65,3.65)[i%2]; y=(1.62,-1.24)[i//2]
            heat=vector_map(values,center=(x,y),cell=.25,color=color)
            self.add_detail(heat,text(f"{name} · ||d||={np.linalg.norm(values):.3f}",
                                      (x,y+1.32),color,20),
                            text(f"đỉnh={np.max(values):.3f}",
                                 (x,y-1.28),WHITE,17),run_time=.65)
            heats.append(heat)
            self.hold(1.5)
        clipped=np.flatnonzero(desc.first_l2>.2)
        self.caption_to(f"{len(clipped)} thành phần vượt 0.2 được cắt trước L2 cuối")
        for clipped_index in clipped[:4]:
            self.play(Indicate(heats[1][int(clipped_index)],color=YELLOW),
                      Indicate(heats[2][int(clipped_index)],color=YELLOW),
                      run_time=.75)
        self.hold(3)
        idx=int(np.argmax(desc.first_l2))
        self.add_detail(text(f"d[{idx}]: {desc.raw[idx]:.3f} → {desc.first_l2[idx]:.3f} "
                             f"→ {desc.clipped[idx]:.3f} → {desc.final[idx]:.3f}",
                             (0,-2.90),YELLOW,20))
        assert desc.first_l2[idx]>.2 and desc.clipped[idx]==.2
        self.caption_to("Clip giảm đỉnh lớn; L2 cuối có thể đưa giá trị vượt 0.2")
        self.hold(5)

    def matching(self):
        d=self.data; m=d.accepted
        self.section("So sánh descriptor giữa hai ảnh",
                     "Phát hiện độc lập trên A và B; chọn gần nhất bằng 128 số")
        self.step(8)
        self.hero_to(d.image,center=(-5.55,.25),width=2.60)
        target=image_mobject(d.transformed,width=2.60).move_to([5.55,.25,0])
        self.add_detail(target,text(f"A · {len(d.described)} bản ghi",
                                    (-5.55,2.15),CYAN,21),
                        text(f"B · {len(d.described_b)} bản ghi",
                             (5.55,2.15),PURPLE,21))
        marks_a=VGroup(*[Dot(image_point(self.hero,d.image.shape,p.row_global,p.col_global),
                             radius=.037,color=CYAN) for p in d.described])
        marks_b=VGroup(*[Dot(image_point(target,d.transformed.shape,p.row_global,p.col_global),
                             radius=.037,color=PURPLE) for p in d.described_b])
        self.add_detail(marks_a,marks_b)
        self.hold(3)
        query=d.described[m.query]; near=d.described_b[m.nearest]
        second=d.described_b[m.second]
        qa=Dot(image_point(self.hero,d.image.shape,query.row_global,query.col_global),
               radius=.08,color=YELLOW)
        qb=Dot(image_point(target,d.transformed.shape,near.row_global,near.col_global),
               radius=.08,color=GREEN)
        q2=Dot(image_point(target,d.transformed.shape,second.row_global,second.col_global),
               radius=.07,color=ORANGE)
        self.add_detail(qa,qb,q2)
        qvec=query.descriptor; nvec=near.descriptor; svec=second.descriptor
        qheat=vector_map(qvec,center=(0,1.05),cell=.17,color=CYAN)
        nheat=vector_map(nvec,center=(0,-.92),cell=.17,color=PURPLE)
        self.add_detail(qheat,nheat,text("128D: A(query)",(0,2.54),CYAN,19),
                        text("128D: B(candidate)",(0,-2.32),PURPLE,19))
        self.hold(3)
        idx=int(np.argmax(np.abs(qvec-nvec)))
        sq=float((qvec[idx]-nvec[idx])**2)
        self.play(Indicate(qheat[idx],color=YELLOW),
                  Indicate(nheat[idx],color=YELLOW),run_time=.7)
        self.caption_to(f"Thành phần {idx}: ({qvec[idx]:.3f} − {nvec[idx]:.3f})² = {sq:.4f}")
        self.hold(2)
        distance=float(np.sqrt(np.sum((qvec-nvec)**2)))
        assert np.isclose(distance,m.d1)
        self.caption_to(f"Cộng 128 bình phương rồi căn: d1 = {distance:.4f}")
        self.hold(2)
        squared=(qvec-nvec)**2
        self.play(Transform(nheat,vector_map(squared,center=(0,-.92),
                                             cell=.17,color=YELLOW)),run_time=.75)
        self.caption_to("Bản đồ sai khác bình phương: tổng các ô là d1²")
        self.hold(2)
        self.play(Transform(nheat,vector_map(svec,center=(0,-.92),cell=.17,
                                             color=ORANGE)),run_time=.8)
        distance2=float(np.sqrt(np.sum((qvec-svec)**2)))
        assert np.isclose(distance2,m.d2)
        self.caption_to(f"Ứng viên thứ hai: d2 = {distance2:.4f}")
        distances=sift.compute_descriptor_distances(qvec,
                 np.stack([p.descriptor for p in d.described_b]))
        nearest=np.argsort(distances,kind="stable")[:3]
        ranking=self.add_detail(text("Khoảng cách nhỏ nhất: "+" < ".join(
                             f"B{int(k)}={distances[k]:.3f}" for k in nearest),
                             (0,-2.79),WHITE,19))
        self.hold(4)
        self.remove_detail(ranking)
        self.add_detail(text(f"d1/d2 = {m.d1:.4f}/{m.d2:.4f} = {m.ratio:.3f} < 0.8",
                             (0,-2.85),GREEN,22))
        self.caption_to("Lowe ratio ACCEPT: descriptor gần nhất vượt trội")
        self.hold(3)
        expected=d.affine@np.array([query.col_global,query.row_global,1.])
        truth=Dot(image_point(target,d.transformed.shape,expected[1],expected[0]),
                  radius=.075,color=ORANGE)
        self.add_detail(Line(qa.get_center(),qb.get_center(),color=GREEN,
                             stroke_width=2.2),truth,
                        text(f"Hậu kiểm affine: lỗi {d.geometry_error:.4f} px",
                             (0,3.16),YELLOW,21))
        self.caption_to("Tọa độ affine chỉ được lộ ra sau quyết định descriptor")
        self.hold(4)

        self.section("Ratio có thể từ chối hoặc chấp nhận sai",
                     "Một truy vấn khác: xóa toàn bộ kết quả ví dụ trước")
        self.hero_to(d.image,center=(-4.8,.25),width=3.1)
        target=image_mobject(d.transformed,width=3.1).move_to([4.8,.25,0])
        self.add_detail(target)
        if d.rejected is not None:
            r=d.rejected
            self.add_detail(text(f"REJECT: {r.d1:.4f}/{r.d2:.4f} = "
                                 f"{r.ratio:.3f} ≥ 0.8",
                                 (0,2.72),MUTED,22))
            self.hold(3)
        matches=sift.match_descriptors(d.described,d.described_b)
        false=[(sift.evaluate_geometric_correspondence(d.described[z.query],
                d.described_b[z.nearest],d.affine)[0],z) for z in matches if z.accepted]
        bad_error,bad=max((item for item in false if item[0]>4),key=lambda x:x[0])
        self.section("ACCEPT vẫn có thể ghép sai hình học",
                     "Ví dụ thứ ba: ratio đúng rule, vị trí ảnh lại lệch")
        self.hero_to(d.image,center=(-4.8,.25),width=3.1)
        target=image_mobject(d.transformed,width=3.1).move_to([4.8,.25,0])
        self.add_detail(target)
        a=d.described[bad.query]; b=d.described_b[bad.nearest]
        source=image_point(self.hero,d.image.shape,a.row_global,a.col_global)
        chosen=image_point(target,d.transformed.shape,b.row_global,b.col_global)
        axy=d.affine@np.array([a.col_global,a.row_global,1.])
        expected=image_point(target,d.transformed.shape,axy[1],axy[0])
        self.add_detail(Dot(source,radius=.085,color=YELLOW),
                        Dot(chosen,radius=.085,color=GREEN),
                        Line(source,chosen,color=GREEN,stroke_width=2.3),
                        text(f"Descriptor: {bad.ratio:.3f} < 0.8 → ACCEPT",
                             (0,2.75),GREEN,22))
        self.hold(2.5)
        self.add_detail(Dot(expected,radius=.085,color=ORANGE),
                        Line(chosen,expected,color=ORANGE,stroke_width=2.3),
                        text(f"Affine hậu kiểm: lệch {bad_error:.2f} px",
                             (0,-2.52),ORANGE,22))
        self.caption_to("ACCEPT theo descriptor không chứng minh đúng hình học")
        self.hold(5)

    def recap(self):
        d=self.data
        self.section("Từ ảnh gốc đến ứng viên ổn định",
                     "Mỗi tầng lấy dữ liệu từ kết quả tầng trước")
        self.step(9)
        first=(("Ảnh gốc",d.image,False),
               ("Gaussian",d.pyramid[0].gaussians[1],False),
               ("DoG",d.pyramid[0].dogs[1],True),
               ("Cực trị 26",d.pyramid[0].dogs[d.candidate.layer],True),
               ("Tinh chỉnh",d.pyramid[0].dogs[d.selected.layer],True))
        self.hero_to(d.image,center=(-6.25,.25),width=1.8)
        for i,(name,array,signed) in enumerate(first):
            x=-5.5+i*2.55
            if i==0:
                self.hero_to(array,center=(x,.15),width=1.7,run_time=.35)
                self.add_detail(text(name,(x,1.45),PURPLE,18),run_time=.2)
            else:
                self.add_detail(image_mobject(array,width=1.7,signed=signed)
                                .move_to([x,.15,0]),
                                text(name,(x,1.45),YELLOW if i>=3 else PURPLE,18),
                                run_time=.35)
            self.hold(.75)
        counts=Counter(p.status for p in d.filtered)
        self.add_detail(text(f"{len(d.raw)} thô → {len(d.raw)-counts['invalid']} "
                             f"tinh chỉnh → {counts['kept']} vị trí",
                             (0,-2.25),GREEN,23))
        self.hold(4)
        self.section("Từ keypoint đến phép so khớp",
                     "Hướng → 16×8 histogram → chuẩn hóa → ratio → hậu kiểm")
        self.hero_to(d.image,center=(-5.6,.2),width=2.6)
        image,j=sift.gaussian_level(d.selected,d.pyramid[d.selected.octave])
        patch,_=sift.sample_oriented_descriptor_patch(d.oriented,
                                                     d.pyramid[d.oriented.octave],48)
        self.add_detail(fixed_image(patch,2.3).move_to([-2.15,.2,0]),
                        vector_map(d.descriptor.raw,center=(1.3,.2),cell=.24),
                        image_mobject(d.transformed,width=2.35).move_to([5.5,.2,0]),
                        text(f"{len(d.described)} descriptor",(-5.6,2.1),CYAN,19),
                        text("Patch theo hướng",(-2.15,2.1),PURPLE,19),
                        text("128D",(1.3,2.1),ORANGE,21),
                        text("Ảnh B + hậu kiểm",(5.5,2.1),GREEN,19))
        self.hold(6)
        self.section("Một vật thể, nhiều cách nhìn",
                     "SIFT hỗ trợ so khớp; hình học kiểm tra correspondence")
        self.hero_to(d.image,center=(-3.3,.25),width=4.0)
        self.add_detail(image_mobject(d.transformed,width=4.0).move_to([3.3,.25,0]))
        self.hold(6)
