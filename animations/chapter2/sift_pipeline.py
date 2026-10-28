"""SIFT teaching pipeline using computed Chapter 2 scale space and descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np
from manim import (
    Arrow, Circle, Create, Dot, FadeIn, FadeOut, Indicate, Line, Rectangle,
    ReplacementTransform, Scene, Transform, Group as VGroup,
)

from chapter2_common import (
    BG, CYAN, GREEN, GRID, MUTED, ORANGE, PURPLE, WHITE, YELLOW,
    TextSwap, check_scene_bounds, framed_image, image_mobject, pixel_grid,
    progress_bar, text,
)
from utils import scale_space, sift


@dataclass(frozen=True)
class SIFTFixture:
    image: np.ndarray
    transformed: np.ndarray
    transform: np.ndarray
    pyramid: tuple
    raw: tuple
    filtered: tuple
    described: tuple
    selected_raw: sift.Keypoint
    selected: sift.Keypoint
    oriented: sift.Keypoint
    orientation: sift.Orientation
    descriptor: sift.Descriptor
    neighbors: tuple
    match_image: np.ndarray
    match_target: np.ndarray
    match_transform: np.ndarray
    match_a: tuple
    match_b: tuple
    match: sift.Match
    match_error: float
    match_correct: bool
    false_match: sift.Match
    false_error: float
    noisy_low: sift.Keypoint
    noisy_edge: sift.Keypoint
    noisy_edge_ratio: float


def build_sift_fixture() -> SIFTFixture:
    image = sift.teaching_presets(129)["Blob và góc rõ"]
    transformed, transform = sift.transform_scene(image, rotation=20, scale=.92)
    pyramid, raw, filtered, described = sift.pipeline(image)
    ranked = sorted((p for p in filtered if p.status == "kept"),
                    key=lambda p: -abs(p.response))
    selected = next(p for p in ranked if p.octave == 0 and abs(p.delta[0]) > .1 and len(
        sift.calculate_orientation_histogram(p, pyramid[p.octave]).peaks) == 1)
    raw_origin = raw[next(i for i, p in enumerate(filtered) if p is selected)]
    oriented = sift.assign_keypoint_orientations(selected, pyramid[selected.octave])[0]
    orientation = sift.calculate_orientation_histogram(selected, pyramid[selected.octave])
    descriptor = sift.build_sift_descriptor(oriented, pyramid[oriented.octave])
    neighbors = scale_space.inspect_neighbors(pyramid[raw_origin.octave].dogs,
                                              raw_origin.layer, raw_origin.row, raw_origin.col)

    match_image = sift.teaching_presets(129)["Cấu trúc bất đối xứng"]
    match_target, matrix = sift.transform_scene(match_image, rotation=20, scale=.92)
    _, _, _, match_a = sift.pipeline(match_image)
    _, _, _, match_b = sift.pipeline(match_target)
    matches = sift.match_descriptors(match_a, match_b, .8)
    good = [(m, *sift.evaluate_geometric_correspondence(match_a[m.query], match_b[m.nearest], matrix))
            for m in matches if m.accepted]
    assert good
    # Prefer an accepted geometrically correct example, selected only after
    # descriptor decisions; ground truth is used for post-hoc explanation.
    chosen = min(good, key=lambda item: item[1])
    false = max((item for item in good if not item[2]), key=lambda item: item[1])
    noisy = sift.teaching_presets(129)["Cảnh nhiễu"]
    noisy_pyramid, _, noisy_filtered, _ = sift.pipeline(noisy, with_descriptors=False)
    low = next(p for p in noisy_filtered if p.status == "low contrast")
    edge = next(p for p in noisy_filtered if p.status == "edge")
    hxy = sift.calculate_spatial_hessian(noisy_pyramid[edge.octave].dogs,
                                         edge.layer, edge.row, edge.col)
    edge_ratio = sift.edge_decision(hxy, 10)[3]
    return SIFTFixture(image, transformed, transform, pyramid, tuple(raw), tuple(filtered),
                       tuple(described), raw_origin, selected, oriented, orientation,
                       descriptor, neighbors, match_image, match_target, matrix,
                       tuple(match_a), tuple(match_b), *chosen,
                       false[0], false[1], low, edge, edge_ratio)


def image_point(mob, shape, row, col):
    h, w = shape
    return np.array([mob.get_left()[0] + (col+.5)*mob.width/w,
                     mob.get_top()[1] - (row+.5)*mob.height/h, 0])


def histogram(values, center, width, height, color, labels=False):
    a=np.asarray(values,dtype=float)
    maximum=max(float(a.max()),1e-12)
    group=VGroup()
    for i,v in enumerate(a):
        bw=width/len(a)*.8
        bh=max(.015,float(v)/maximum*height)
        x=center[0]-width/2+(i+.5)*width/len(a)
        y=center[1]-height/2+bh/2
        group.add(Rectangle(width=bw,height=bh,fill_color=color,fill_opacity=.9,
                            stroke_width=0).move_to([x,y,0]))
    if labels:
        group.add(text("0°",(center[0]-width/2,center[1]-height/2-.27),MUTED,18))
        group.add(text("360°",(center[0]+width/2,center[1]-height/2-.27),MUTED,18))
    return group


class SIFTFeaturePipeline(Scene):
    def play(self,*animations,**kwargs):
        super().play(*animations,**kwargs)
        check_scene_bounds(self)

    def cue(self,message,color=WHITE):
        new=text(message,(0,-3.43),color,27)
        old=self.caption
        self.caption=new
        return TextSwap(old,new)

    def step(self,index):
        return self.marker.animate.move_to([self.step_x[index],-4.29,0])

    def clear_panel(self):
        if self.panel:
            items=self.panel
            self.panel=VGroup()
            return FadeOut(items)
        return None

    def construct(self):
        self.data=build_sift_fixture()
        self.title=text("SIFT · detect, orient, describe, match",(0,3.92),WHITE,35)
        self.caption=text("Can one local feature survive a change of scale and angle?",(0,-3.43),WHITE,27)
        self.steps,self.step_x=progress_bar(["Image","Scale","DoG","Extrema","Refine",
                                             "Orient","128D","Match","Overview"])
        self.marker=Dot([self.step_x[0],-4.29,0],radius=.045,color=CYAN)
        self.add(self.title,self.caption,self.steps,self.marker)
        self.panel=VGroup()
        self.motivation()
        self.scale_space()
        self.dog()
        self.extremum()
        self.localization()
        self.filtering()
        self.orientation()
        self.rotate_frame()
        self.descriptor()
        self.normalize()
        self.matching()
        self.geometry()
        self.overview()

    def motivation(self):
        a=framed_image(self.data.image,"Original",(-3.4,.15,0),width=4.6,accent=CYAN)
        b=framed_image(self.data.transformed,"Rotated 20° · scale 0.92",(3.4,.15,0),width=4.6,accent=PURPLE)
        self.main=a
        self.play(FadeIn(a),run_time=.65)
        self.play(FadeIn(b),run_time=.65)
        self.panel=VGroup(b)
        self.wait(4.35)

    def scale_space(self):
        p=self.data.pyramid
        layers=VGroup(*[framed_image(p[0].gaussians[j],f"σ={p[0].local_sigmas[j]:.2f}",
                                     (x,.9,0),width=2.35,accent=PURPLE,title_size=20)
                         for j,x in enumerate((-3.9,-1.3,1.3,3.9))])
        octave=framed_image(p[1].gaussians[0],"octave 1 · 65×65",(5.8,-1.75,0),width=1.8,
                            accent=PURPLE,title_size=19)
        equation=text("L(x,y,σ) = G(x,y,σ) * I(x,y)",(0,-1.6),WHITE,29)
        sigmas=text(f"σ_global = {p[1].local_sigmas[0]:.2f} × 2¹ = {p[1].global_sigmas[0]:.2f}",
                    (0,-2.55),PURPLE,26)
        self.play(self.clear_panel(),Transform(self.main,layers[0]),self.step(1),
                  self.cue("Increasing σ removes progressively finer detail"),run_time=.75)
        self.play(*[FadeIn(layer) for layer in layers[1:]],FadeIn(equation),run_time=.85)
        self.play(FadeIn(octave),FadeIn(sigmas),run_time=.55)
        self.wait(5.3)
        self.panel=VGroup(*layers[1:],octave,equation,sigmas)

    def dog(self):
        p=self.data.pyramid[0]
        low=framed_image(p.gaussians[1],f"L(σ={p.local_sigmas[1]:.2f})",(-4,.35,0),width=3.4,
                          accent=CYAN,title_size=22)
        high=framed_image(p.gaussians[2],f"L(σ={p.local_sigmas[2]:.2f})",(0,.35,0),width=3.4,
                           accent=PURPLE,title_size=22)
        result=framed_image(p.dogs[1],"Signed DoG",(4,.35,0),width=3.4,signed=True,
                             accent=ORANGE,title_size=23)
        subtract=text("DoG = L(kσ) − L(σ)",(0,-2.05),WHITE,29)
        signs=text("blue: negative      orange: positive",(0,-2.65),MUTED,23)
        self.play(self.clear_panel(),Transform(self.main,low),self.step(2),
                  self.cue("Subtract consecutive Gaussian layers pixel by pixel"),run_time=.75)
        self.play(FadeIn(high),FadeIn(subtract),run_time=.5)
        self.play(FadeIn(result),FadeIn(signs),run_time=.65)
        self.wait(5.3)
        self.panel=VGroup(high,result,subtract,signs)

    def extremum(self):
        d=self.data; kp=d.selected_raw; octave=d.pyramid[kp.octave]
        panels=VGroup(*[framed_image(octave.dogs[j],f"DoG layer {j}",(x,.65,0),width=3.0,
                                     signed=True,accent=ORANGE,title_size=22)
                         for j,x in zip((kp.layer-1,kp.layer,kp.layer+1),(-4.25,0,4.25))])
        marks=VGroup(*[Circle(radius=.10,color=YELLOW,stroke_width=3).move_to(
            image_point(panel[0],octave.dogs[0].shape,kp.row,kp.col)) for panel in panels])
        for mark in marks: mark.set_z_index(3)
        values=[e.value for e in d.neighbors]
        strict=all(kp.dog_value>v for v in values) or all(kp.dog_value<v for v in values)
        evidence=text(f"p={kp.dog_value:+.5f}  ·  8+9+9=26 neighbors",(0,-1.85),YELLOW,27)
        verdict=text(f"strict {kp.kind} · DoG candidate",(0,-2.55),GREEN if strict else ORANGE,27)
        self.play(self.clear_panel(),Transform(self.main,panels[0]),self.step(3),
                  self.cue("Compare a candidate across three adjacent DoG layers"),run_time=.8)
        self.play(FadeIn(panels[1:]),FadeIn(marks),FadeIn(evidence),run_time=.7)
        self.play(*[Indicate(mark,color=YELLOW) for mark in marks],FadeIn(verdict),run_time=.7)
        local=VGroup(*[pixel_grid(1000*octave.dogs[j][kp.row-1:kp.row+2,
                                                      kp.col-1:kp.col+2],
                                     [x,.5,0],cell=.75,decimals=1,signed=True,font_size=18)
                       for j,x in zip((kp.layer-1,kp.layer,kp.layer+1),(-4.25,0,4.25))])
        local_titles=VGroup(*[text(f"layer {j}",(x,2.25),PURPLE,23)
                              for j,x in zip((kp.layer-1,kp.layer,kp.layer+1),(-4.25,0,4.25))])
        local_note=text("3 × 3 × 3 DoG values shown as 1000 × DoG",(0,-1.15),MUTED,22)
        center=Circle(radius=.34,color=YELLOW,stroke_width=3).move_to(local[1][4])
        self.play(FadeOut(self.main),FadeOut(panels[1:]),FadeOut(marks),
                  FadeIn(local),FadeIn(local_titles),FadeIn(local_note),Create(center),
                  self.cue("One center must exceed all 26 distinct neighbors"),run_time=.8)
        self.play(Indicate(local[0],color=YELLOW),Indicate(VGroup(*local[1][:4],*local[1][5:]),color=YELLOW),
                  Indicate(local[2],color=YELLOW),run_time=.8)
        self.wait(6.35)
        self.panel=VGroup(local,local_titles,local_note,center,evidence,verdict)

    def localization(self):
        d=self.data; kp=d.selected; original=d.selected_raw
        t=sift.calculate_dog_derivatives(d.pyramid[kp.octave].dogs,kp.layer,kp.row,kp.col)
        lattice=text(f"refined grid (x,y,s)=({kp.col},{kp.row},{kp.layer})",(-3.6,2.15),CYAN,23)
        equation=text("H δ = −g   ·   H is 3×3 in (x,y,scale)",(2.65,2.15),WHITE,24)
        delta=text(f"δ = ({kp.delta[0]:+.3f}, {kp.delta[1]:+.3f}, {kp.delta[2]:+.3f})",
                   (2.0,.75),YELLOW,28)
        response=text(f"interpolated DoG = {kp.response:+.5f}",(2,-.15),ORANGE,27)
        marker=Dot([-3.2,.2,0],radius=.12,color=YELLOW)
        grid=VGroup(*[Line([-4.2+i*.5,-.8,0],[-4.2+i*.5,1.2,0],color=GRID)
                       for i in range(5)],*[Line([-4.2,-.8+i*.5,0],[-2.2,-.8+i*.5,0],color=GRID)
                                      for i in range(5)])
        self.play(self.clear_panel(),FadeOut(self.main),FadeIn(grid),FadeIn(marker),
                  FadeIn(lattice),FadeIn(equation),self.step(4),
                  self.cue("Taylor refinement moves the candidate off the integer grid"),run_time=.8)
        self.play(marker.animate.shift([kp.delta[0]*.5,-kp.delta[1]*.5,0]),
                  FadeIn(delta),FadeIn(response),run_time=.85)
        self.wait(6.3)
        self.panel=VGroup(grid,marker,lattice,equation,delta,response)

    def filtering(self):
        d=self.data
        from collections import Counter
        counts=Counter(p.status for p in d.filtered)
        title=text("Filter the refined candidates",(0,2.45),WHITE,30)
        contrast=text("|interpolated DoG| ≥ 0.008",(-3.35,1.0),ORANGE,27)
        edge=text("Spatial Hessian 2×2 · limit 12.1",(3.25,1.0),PURPLE,25)
        examples=text(f"noisy fixture: |D|={abs(d.noisy_low.response):.4f} REJECT"
                      f"    ·    edge ratio={d.noisy_edge_ratio:.1f} REJECT",
                      (0,.2),YELLOW,23)
        limits=text(f"{len(d.raw)} raw → {counts['kept']} kept  ·  {counts['invalid']} invalid"
                    f"  ·  {counts['edge']} edge  ·  {counts['low contrast']} low contrast",
                    (0,-.6),GREEN,24)
        note=text("Contrast uses grayscale [0,1] DoG units",(0,-1.75),MUTED,23)
        self.play(self.clear_panel(),FadeIn(title),FadeIn(contrast),FadeIn(edge),
                  self.cue("Contrast and edge filtering use distinct tests"),run_time=.75)
        self.play(FadeIn(examples),FadeIn(limits),FadeIn(note),run_time=.55)
        self.wait(5.35)
        self.panel=VGroup(title,contrast,edge,examples,limits,note)

    def orientation(self):
        d=self.data; info=d.orientation; kp=d.selected
        image,_=sift.gaussian_level(kp,d.pyramid[kp.octave])
        crop=image[max(0,kp.row-14):kp.row+15,max(0,kp.col-14):kp.col+15]
        patch=framed_image(crop,"Local gradients",(-4.4,.35,0),width=3.45,accent=CYAN,title_size=23)
        hist=histogram(info.histogram,(2.45,.45),width=6.8,height=2.6,color=PURPLE,labels=True)
        heading=text("36 orientation bins · weighted gradient votes",(2.45,2.35),PURPLE,25)
        peak=text(f"dominant orientation: {info.peaks[0]:.0f}°",(2.45,-1.65),YELLOW,27)
        secondary=text(f"{len(info.peaks)} orientation record(s) at this one location",(2.45,-2.3),MUTED,23)
        self.play(self.clear_panel(),FadeIn(patch),FadeIn(heading),self.step(5),
                  self.cue("Each gradient votes by magnitude and Gaussian spatial weight"),run_time=.75)
        samples=sorted(info.contributions,key=lambda s:-s[5])[:5]
        sample_marks=VGroup()
        for row,col,mag,angle,spatial,weight in samples:
            mark=Dot(image_point(patch[0],crop.shape,row-max(0,kp.row-14),col-max(0,kp.col-14)),
                     radius=.065,color=YELLOW)
            if np.all(np.isfinite(mark.get_center())):
                sample_marks.add(mark)
        self.play(FadeIn(sample_marks),FadeIn(hist),run_time=.75)
        for mark,(_,_,_,angle,_,_) in zip(sample_marks,samples[:3]):
            low=int(np.floor(angle/10))%36
            self.play(Indicate(mark,color=YELLOW),Indicate(hist[low],color=YELLOW),
                      Indicate(hist[(low+1)%36],color=YELLOW),run_time=.4)
        self.play(FadeIn(peak),FadeIn(secondary),Indicate(hist,color=YELLOW),run_time=.7)
        self.wait(5.3)
        self.panel=VGroup(patch,heading,sample_marks,hist,peak,secondary)

    def rotate_frame(self):
        d=self.data; kp=d.oriented
        image,_=sift.gaussian_level(kp,d.pyramid[kp.octave])
        crop=image[max(0,kp.row-20):kp.row+21,max(0,kp.col-20):kp.col+21]
        patch=framed_image(crop,"Same image region",(-3.7,.25,0),width=4.1,accent=CYAN)
        center=patch[0].get_center()
        theta=radians(kp.orientation)
        axes=VGroup(
            Arrow(center,center+1.65*np.array([cos(theta),-sin(theta),0]),buff=0,color=YELLOW),
            Arrow(center,center+1.25*np.array([-sin(theta),-cos(theta),0]),buff=0,color=PURPLE))
        label=text(f"local frame aligned to {kp.orientation:.0f}°",(2.8,1.2),YELLOW,28)
        convention=text("image x → right    image y → down",(2.8,.2),MUTED,24)
        sigma=text(f"support width = 3σ_local = {3*kp.sigma_local:.2f} px per cell",
                   (2.8,-1.15),PURPLE,23)
        self.play(self.clear_panel(),FadeIn(patch),FadeIn(axes),FadeIn(label),
                  self.cue("Align the descriptor frame with the assigned orientation"),run_time=.8)
        self.play(FadeIn(convention),FadeIn(sigma),run_time=.45)
        self.wait(4.2)
        self.panel=VGroup(patch,axes,label,convention,sigma)

    def descriptor(self):
        d=self.data; desc=d.descriptor
        cells=VGroup()
        for row in range(4):
            for col in range(4):
                pos=[-3.8+(col-1.5)*.95,.4+(1.5-row)*.95,0]
                energy=float(desc.cells[row,col].sum())
                alpha=.18+.72*energy/max(float(desc.cells.sum()),1e-12)*8
                cells.add(Rectangle(width=.9,height=.9,stroke_color=GRID,stroke_width=1.5,
                                    fill_color=PURPLE,fill_opacity=min(alpha,.85)).move_to(pos))
        label=text("4 × 4 cells · 8 orientation bins each",(-3.8,2.8),WHITE,25)
        vector=histogram(desc.raw,(2.85,.15),width=8.2,height=2.5,color=ORANGE)
        dimension=text("4 × 4 × 8 = 128 dimensions",(2.85,-1.8),YELLOW,28)
        sample=max((s for s in desc.samples if 0<=round(s[2])<4 and 0<=round(s[3])<4),
                   key=lambda s:s[5])
        row,col,u,v,angle,weight=sample
        rr=int(np.clip(round(v),0,3)); cc=int(np.clip(round(u),0,3)); bb=int(np.floor(angle/45))%8
        idx=sift.descriptor_index(rr,cc,bb)
        sample_dot=Dot([-3.8+(u-1.5)*.95,.4+(1.5-v)*.95,0],radius=.065,color=YELLOW)
        detail=text(f"pixel [{row},{col}], vote={weight:.3f} → cell [{rr},{cc}], bin {bb}",
                    (0,-2.3),CYAN,23)
        component=text(f"raw vector[{idx}] = {desc.raw[idx]:.3f}",(0,-2.73),YELLOW,23)
        self.play(self.clear_panel(),FadeIn(cells),FadeIn(label),self.step(6),
                  self.cue("Gradient samples contribute to nearby cells and orientation bins"),run_time=.8)
        selected=cells[rr*4+cc].copy().set_fill(opacity=0).set_stroke(YELLOW,3)
        self.play(FadeIn(selected),FadeIn(sample_dot),FadeIn(detail),run_time=.5)
        self.play(FadeIn(vector),FadeIn(dimension),FadeIn(component),
                  Indicate(vector[idx],color=YELLOW),run_time=.8)
        self.wait(6.5)
        self.panel=VGroup(cells,label,selected,sample_dot,detail,component,vector,dimension)

    def normalize(self):
        d=self.data.descriptor
        names=("raw", "first L2", "clip at 0.2", "final L2")
        arrays=(d.raw,d.first_l2,d.clipped,d.final)
        displays=VGroup(*[histogram(a,(-4.4+2.95*i,.25),2.55,2.4,
                                        (ORANGE,CYAN,PURPLE,GREEN)[i])
                          for i,a in enumerate(arrays)])
        headings=VGroup(*[text(name,(-4.4+2.95*i,2.15),(ORANGE,CYAN,PURPLE,GREEN)[i],23)
                          for i,name in enumerate(names)])
        example=int(np.argmax(d.first_l2))
        values=text(f"component {example}: {d.raw[example]:.3f} → {d.first_l2[example]:.3f}"
                    f" → {d.clipped[example]:.3f} → {d.final[example]:.3f}",
                    (0,-1.9),YELLOW,25)
        note=text("Final values may exceed 0.2 after the second normalization",(0,-2.55),MUTED,23)
        self.play(self.clear_panel(),FadeIn(displays[0]),FadeIn(headings[0]),
                  self.cue("Normalization reduces sensitivity to some intensity changes"),run_time=.7)
        for i in range(1,4):
            self.play(FadeIn(displays[i]),FadeIn(headings[i]),run_time=.45)
        self.play(FadeIn(values),FadeIn(note),run_time=.55)
        self.wait(5.25)
        self.panel=VGroup(displays,headings,values,note)

    def matching(self):
        d=self.data; m=d.match
        left=framed_image(d.match_image,"A · query",(-3.75,.25,0),width=4.1,accent=CYAN)
        right=framed_image(d.match_target,"B · candidates",(3.75,.25,0),width=4.1,accent=PURPLE)
        a=d.match_a[m.query]; b=d.match_b[m.nearest]
        qa=Dot(image_point(left[0],d.match_image.shape,a.row_global,a.col_global),
               radius=.085,color=YELLOW)
        qb=Dot(image_point(right[0],d.match_target.shape,b.row_global,b.col_global),
               radius=.085,color=GREEN)
        line=Line(qa.get_center(),qb.get_center(),color=GREEN,stroke_width=3)
        distances=text(f"d₁={m.d1:.3f}   d₂={m.d2:.3f}   d₁/d₂={m.ratio:.3f}",
                       (0,-2.22),YELLOW,28)
        verdict=text(f"{m.ratio:.3f} < 0.8  →  ACCEPT",(0,-2.72),GREEN,27)
        self.play(self.clear_panel(),FadeIn(left),FadeIn(right),self.step(7),
                  self.cue("Find the nearest and second-nearest real 128D descriptors"),run_time=.75)
        self.play(FadeIn(qa),FadeIn(qb),FadeIn(distances),run_time=.6)
        self.play(Create(line),FadeIn(verdict),run_time=.65)
        self.wait(4.25)
        self.panel=VGroup(left,right,qa,qb,line,distances,verdict)

    def geometry(self):
        d=self.data; m=d.match
        source=d.match_a[m.query]
        target=d.match_b[m.nearest]
        expected=d.match_transform@np.array([source.col_global,source.row_global,1.])
        right=self.panel[1]
        expected_dot=Dot(image_point(right[0],d.match_target.shape,expected[1],expected[0]),
                         radius=.055,color=ORANGE)
        error=text(f"post-hoc affine error = {d.match_error:.2f} px",
                   (0,3.15),GREEN if d.match_correct else ORANGE,24)
        self.play(FadeIn(expected_dot),FadeIn(error),self.cue("Now compare with the known affine transform",ORANGE),
                  run_time=.6)
        self.wait(2.0)
        false=d.false_match
        source_false=d.match_a[false.query]; target_false=d.match_b[false.nearest]
        left=self.panel[0]
        qa=self.panel[2]; qb=self.panel[3]; line=self.panel[4]
        false_a=image_point(left[0],d.match_image.shape,source_false.row_global,source_false.col_global)
        false_b=image_point(right[0],d.match_target.shape,target_false.row_global,target_false.col_global)
        false_expected=d.match_transform@np.array([source_false.col_global,source_false.row_global,1.])
        false_expected_pos=image_point(right[0],d.match_target.shape,false_expected[1],false_expected[0])
        false_ratio=text(f"ratio={false.ratio:.3f} < 0.8   ·   geometric error={d.false_error:.1f} px",
                         (0,3.15),ORANGE,23)
        old_distances=self.panel[5];old_verdict=self.panel[6]
        new_distances=text(f"d₁={false.d1:.3f}   d₂={false.d2:.3f}   d₁/d₂={false.ratio:.3f}",
                           (0,-2.22),YELLOW,28)
        new_verdict=text(f"{false.ratio:.3f} < 0.8  →  ACCEPT",(0,-2.72),ORANGE,27)
        self.play(qa.animate.move_to(false_a),qb.animate.move_to(false_b),
                  Transform(line,Line(false_a,false_b,color=ORANGE,stroke_width=3)),
                  expected_dot.animate.move_to(false_expected_pos),TextSwap(error,false_ratio),
                  TextSwap(old_distances,new_distances),TextSwap(old_verdict,new_verdict),
                  self.cue("An accepted descriptor match can still be geometrically wrong",ORANGE),run_time=.8)
        self.wait(2.2)
        self.panel.remove(old_distances,old_verdict)
        self.panel.add(expected_dot,false_ratio,new_distances,new_verdict)

    def overview(self):
        names=["Image","Gaussian","DoG","Extrema","Taylor","Filter","Orient","128D","Match"]
        boxes=VGroup()
        for i,name in enumerate(names):
            row,col=divmod(i,3)
            x=(col-1)*4.65; y=1.7-row*1.55
            rect=Rectangle(width=3.75,height=.95,color=GRID,stroke_width=2,
                           fill_color=BG,fill_opacity=.9).move_to([x,y,0])
            label=text(name,(x,y),GREEN if i==8 else PURPLE if i in (1,6) else CYAN,26)
            boxes.add(VGroup(rect,label))
        summary=text("Scale + orientation + description give approximate robustness",(0,-2.9),WHITE,25)
        self.play(self.clear_panel(),FadeIn(boxes),self.step(8),
                  self.cue("SIFT combines detection, scale, orientation and description",GREEN),run_time=.85)
        self.play(FadeIn(summary),run_time=.45)
        self.wait(4.6)
