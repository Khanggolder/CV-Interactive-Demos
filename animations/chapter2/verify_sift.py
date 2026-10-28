"""Independent checks for displayed SIFT data and full video decoding."""

from __future__ import annotations

import argparse
from collections import Counter

import cv2
import numpy as np

from chapter2_common import HERE
from sift_pipeline import SIFTFeaturePipeline, build_sift_fixture
from utils import sift

REVIEW=HERE/"media/videos/sift_pipeline/480p15/SIFTFeaturePipeline_review_v5.mp4"
FINAL=HERE/"media/videos/sift_pipeline/1080p60/SIFTFeaturePipeline_final_v4.mp4"


def validate():
    d=build_sift_fixture()
    assert d.image.shape==(129,129) and np.isfinite(d.image).all()
    assert len(d.pyramid)==3 and all(len(o.gaussians)==6 and len(o.dogs)==5 for o in d.pyramid)
    for octave in d.pyramid:
        for j in range(5):
            np.testing.assert_allclose(octave.dogs[j],octave.gaussians[j+1]-octave.gaussians[j])
            assert np.isclose(octave.global_sigmas[j],octave.local_sigmas[j]*2**octave.index)
        if octave.index:
            prior=d.pyramid[octave.index-1]
            np.testing.assert_allclose(octave.gaussians[0],prior.gaussians[3][::2,::2])
    k=d.selected_raw; stack=d.pyramid[k.octave].dogs
    blocks=[stack[j][k.row-1:k.row+2,k.col-1:k.col+2].ravel()
            for j in (k.layer-1,k.layer,k.layer+1)]
    neighbors=np.r_[blocks[0],np.delete(blocks[1],4),blocks[2]]
    assert len(neighbors)==26 and len(d.neighbors)==26
    assert (np.all(k.dog_value>neighbors) if k.kind=="MAX" else np.all(k.dog_value<neighbors))
    np.testing.assert_allclose(sorted(neighbors),sorted(e.value for e in d.neighbors))
    # Refinement may step to another integer site before its final Taylor solve.
    refined=d.selected
    at=lambda ds,dy,dx: float(stack[refined.layer+ds][refined.row+dy,refined.col+dx])
    g=np.array([(at(0,0,1)-at(0,0,-1))/2,
                (at(0,1,0)-at(0,-1,0))/2,
                (at(1,0,0)-at(-1,0,0))/2])
    h=np.array([[at(0,0,1)+at(0,0,-1)-2*at(0,0,0),
                 (at(0,1,1)-at(0,1,-1)-at(0,-1,1)+at(0,-1,-1))/4,
                 (at(1,0,1)-at(1,0,-1)-at(-1,0,1)+at(-1,0,-1))/4],
                [0,at(0,1,0)+at(0,-1,0)-2*at(0,0,0),
                 (at(1,1,0)-at(1,-1,0)-at(-1,1,0)+at(-1,-1,0))/4],
                [0,0,at(1,0,0)+at(-1,0,0)-2*at(0,0,0)]])
    h[1,0]=h[0,1];h[2,0]=h[0,2];h[2,1]=h[1,2]
    assert h.shape==(3,3) and np.allclose(h,h.T)
    np.testing.assert_allclose(np.linalg.solve(h,-g),d.selected.delta,atol=1e-8)
    assert np.isclose(d.selected.response,at(0,0,0)+.5*g@np.array(d.selected.delta))
    assert abs(d.selected.response)>=.008
    hxy=sift.calculate_spatial_hessian(stack,d.selected.layer,d.selected.row,d.selected.col)
    assert hxy.shape==(2,2) and sift.edge_decision(hxy,10)[0]
    assert Counter(x.status for x in d.filtered)["kept"]==7
    assert len(d.described)==23
    assert d.orientation.histogram.shape==(36,)
    raw_hist=np.zeros(36)
    for _,_,mag,angle,spatial,weight in d.orientation.contributions:
        assert np.isclose(weight,mag*spatial)
        position=angle/10; low=int(np.floor(position))%36; fraction=position-np.floor(position)
        raw_hist[low]+=weight*(1-fraction)
        raw_hist[(low+1)%36]+=weight*fraction
    np.testing.assert_allclose(raw_hist,d.orientation.raw_histogram)
    smooth=(np.roll(raw_hist,2)+4*np.roll(raw_hist,1)+6*raw_hist+
            4*np.roll(raw_hist,-1)+np.roll(raw_hist,-2))/16
    np.testing.assert_allclose(smooth,d.orientation.histogram)
    assert d.oriented.orientation in d.orientation.peaks
    assert len(sift.assign_keypoint_orientations(d.selected,d.pyramid[d.selected.octave]))==len(d.orientation.peaks)
    desc=d.descriptor
    assert desc.cells.shape==(4,4,8) and desc.raw.shape==(128,)
    np.testing.assert_allclose(desc.raw,desc.cells.reshape(128))
    independently_accumulated=np.zeros((4,4,8))
    for _,_,u,v,angle,weight in desc.samples:
        b=angle/45
        for rr in (int(np.floor(v)),int(np.floor(v))+1):
            for cc in (int(np.floor(u)),int(np.floor(u))+1):
                for bb in (int(np.floor(b)),int(np.floor(b))+1):
                    if 0<=rr<4 and 0<=cc<4:
                        independently_accumulated[rr,cc,bb%8]+=(
                            weight*(1-abs(v-rr))*(1-abs(u-cc))*(1-abs(b-bb)))
    np.testing.assert_allclose(independently_accumulated,desc.cells,atol=1e-12)
    for r in range(4):
        for c in range(4):
            for b in range(8):
                assert sift.descriptor_index(r,c,b)==(r*4+c)*8+b
    norm=np.linalg.norm(desc.raw)
    np.testing.assert_allclose(desc.first_l2,desc.raw/norm)
    np.testing.assert_allclose(desc.clipped,np.minimum(desc.first_l2,.2))
    np.testing.assert_allclose(desc.final,desc.clipped/np.linalg.norm(desc.clipped))
    described=next(p for p in d.described if p.octave==d.oriented.octave and
                   np.isclose(p.row_global,d.oriented.row_global) and
                   np.isclose(p.col_global,d.oriented.col_global) and
                   p.orientation==d.oriented.orientation)
    np.testing.assert_allclose(described.descriptor,desc.final)
    m=d.match
    assert m.accepted and m.d1 is not None and m.d2 is not None
    q=d.match_a[m.query].descriptor
    distances=np.array([np.sqrt(np.sum((q-p.descriptor)**2)) for p in d.match_b])
    ordered=np.argsort(distances,kind="stable")
    assert (m.nearest,m.second)==tuple(map(int,ordered[:2]))
    assert np.isclose(m.d1,distances[m.nearest]) and np.isclose(m.d2,distances[m.second])
    assert np.isclose(m.ratio,m.d1/m.d2) and (m.ratio<.8)==m.accepted
    point=d.match_a[m.query]; matched=d.match_b[m.nearest]
    expected=d.match_transform@np.array([point.col_global,point.row_global,1.])
    error=np.linalg.norm(expected-np.array([matched.col_global,matched.row_global]))
    assert np.isclose(error,d.match_error) and (error<=4)==d.match_correct
    assert d.noisy_low.status=="low contrast" and abs(d.noisy_low.response)<.008
    assert d.noisy_edge.status=="edge" and d.noisy_edge_ratio>12.1
    fm=d.false_match
    false_q=d.match_a[fm.query]; false_t=d.match_b[fm.nearest]
    false_dist=np.array([np.linalg.norm(false_q.descriptor-p.descriptor) for p in d.match_b])
    assert fm.accepted and np.isclose(fm.ratio,np.sort(false_dist)[0]/np.sort(false_dist)[1])
    false_expected=d.match_transform@np.array([false_q.col_global,false_q.row_global,1.])
    assert np.isclose(d.false_error,np.linalg.norm(false_expected-
                       np.array([false_t.col_global,false_t.row_global]))) and d.false_error>4
    print("SIFT numeric OK: Gaussian/DoG, 26 neighbors, 3x3 Taylor, 2x2 edge filter, 36 bins, 128D, L2/clip/L2, ratio, affine error")


def preflight():
    from manim import tempconfig
    with tempconfig({"skip_animations":True,"save_last_frame":True,"quality":"low_quality"}):
        scene=SIFTFeaturePipeline();scene.render()
        assert not any(m.updaters for root in scene.mobjects for m in root.get_family())
    print("SIFT layout preflight OK: 16:9, finite in-frame objects, no updaters")


def video(path,contact=False):
    import av
    from PIL import Image,ImageDraw
    assert path.is_file() and path.stat().st_size>10000,path
    selected=[];count=0;dark_run=0;longest_dark=0
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        fps=float(stream.average_rate);dims=(stream.width,stream.height)
        for frame in container.decode(video=0):
            count+=1
            if count%max(1,round(fps*7))==1:
                selected.append((count/fps,frame.to_image().copy()))
            gray=cv2.cvtColor(frame.to_ndarray(format="rgb24"),cv2.COLOR_RGB2GRAY)
            dark_run=dark_run+1 if float(np.std(gray))<1.5 else 0
            longest_dark=max(longest_dark,dark_run)
    duration=count/fps
    assert dims==((854,480) if path==REVIEW else (1920,1080))
    assert abs(fps-(15 if path==REVIEW else 60))<.01
    assert 78<=duration<=105,(duration,path)
    assert longest_dark/fps<2
    if contact:
        cols=3;rows=(len(selected)+cols-1)//cols
        sheet=Image.new("RGB",(cols*427,rows*270),"#0B1020")
        draw=ImageDraw.Draw(sheet)
        for i,(time,img) in enumerate(selected):
            x=i%cols*427;y=i//cols*270
            sheet.paste(img.resize((427,240)),(x,y))
            draw.text((x+8,y+243),f"{time:.1f}s",fill="white")
        out=HERE/"media/sift_review_contact.png";sheet.save(out);print("contact",out)
    print(f"SIFT video OK: {path}, {duration:.2f}s, {fps:g} fps, {count} decoded frames, {dims}, {path.stat().st_size} bytes")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--video",choices=("review","final"))
    parser.add_argument("--contact-sheet",action="store_true")
    args=parser.parse_args()
    validate()
    if args.preflight:preflight()
    if args.video:video(REVIEW if args.video=="review" else FINAL,args.contact_sheet)
