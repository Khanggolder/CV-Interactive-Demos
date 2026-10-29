"""Independent numeric, layout and complete-stream video checks for SIFT V2."""

from __future__ import annotations

import argparse
from collections import Counter
from math import sqrt
from pathlib import Path

import cv2
import numpy as np

from sift_deep_dive_data import (build_data, camera_image, dog_examples,
                                 gaussian_pixel_example, signed_pixel)
from utils import scale_space, sift

HERE = Path(__file__).resolve().parent
REVIEW = HERE / "media/videos/sift_deep_dive/480p15/SIFTDeepDive_review.mp4"
FINAL = HERE / "media/videos/sift_deep_dive/1080p30/SIFTDeepDive_final.mp4"


def validate():
    d = build_data()
    assert d.image.shape == (129,129) and d.image.dtype == np.float64
    assert np.array_equal(d.image,camera_image()) and np.isfinite(d.image).all()
    assert 0 <= d.image.min() < d.image.max() <= 1
    assert len(d.pyramid)==3 and len(d.pyramid_b)==3
    for o,octave in enumerate(d.pyramid):
        shape=(129//(2**o)+int(o>0),)*2
        assert len(octave.gaussians)==6 and len(octave.dogs)==5
        assert all(a.shape==shape for a in octave.gaussians+octave.dogs)
        for j in range(6):
            expected=1.6*2**(j/3)
            assert np.isclose(octave.local_sigmas[j],expected)
            assert np.isclose(octave.global_sigmas[j],expected*2**o)
            if j:
                inc=sqrt(octave.local_sigmas[j]**2-octave.local_sigmas[j-1]**2)
                assert np.isclose(octave.increments[j],inc)
                np.testing.assert_allclose(octave.gaussians[j],
                    scale_space.gaussian_blur(octave.gaussians[j-1],inc),atol=1e-12)
        if o:
            np.testing.assert_array_equal(octave.gaussians[0],
                d.pyramid[o-1].gaussians[3][::2,::2])
        for j in range(5):
            np.testing.assert_array_equal(octave.dogs[j],
                octave.gaussians[j+1]-octave.gaussians[j])
    patch,kernel,total=gaussian_pixel_example(d)
    assert kernel.shape==(15,15) and np.isclose(kernel.sum(),1)
    assert np.isclose(total,np.sum(patch*kernel),atol=1e-14)
    assert np.isclose(total,d.pyramid[0].gaussians[0][50,35],atol=1e-10)
    for r,c,lo,hi,diff in dog_examples(d):
        assert np.isclose(diff,hi-lo)
    for o in d.pyramid:
        for j in range(5):
            values=o.dogs[j]
            for r,c in ((20,20),(65//(2**o.index),)*2):
                if r<values.shape[0] and c<values.shape[1]:
                    value,pixel=signed_pixel(d,o.index,j,r,c)
                    assert np.isclose(value,values[r,c])
                    if value>.01: assert pixel[0]>11
                    if value<-.01: assert pixel[2]>32
    assert len(d.raw)==sum(len(scale_space.detect_extrema(o.dogs)) for o in d.pyramid)
    kp=d.candidate; stack=d.pyramid[kp.octave].dogs
    events=scale_space.inspect_neighbors(stack,kp.layer,kp.row,kp.col)
    assert len(events)==26
    neighbor=np.array([e.value for e in events])
    assert (np.all(kp.dog_value>neighbor) if kp.kind=="MAX"
            else np.all(kp.dog_value<neighbor))
    independent=tuple((o,layer,r,c) for o,octave in enumerate(d.pyramid)
                      for layer in range(1,4)
                      for r in range(1,octave.dogs[0].shape[0]-1)
                      for c in range(1,octave.dogs[0].shape[1]-1)
                      if (lambda v,n: np.all(v>n) or np.all(v<n))(
                          octave.dogs[layer][r,c],np.concatenate((
                              octave.dogs[layer-1][r-1:r+2,c-1:c+2].ravel(),
                              np.delete(octave.dogs[layer][r-1:r+2,c-1:c+2].ravel(),4),
                              octave.dogs[layer+1][r-1:r+2,c-1:c+2].ravel()))))
    assert set(independent)==set((p.octave,p.layer,p.row,p.col) for p in d.raw)
    assert d.selected.status=="kept"
    assert (kp.col,kp.row,kp.layer)!=(d.selected.col,d.selected.row,d.selected.layer)
    t=sift.calculate_dog_derivatives(stack,d.selected.layer,d.selected.row,d.selected.col)
    np.testing.assert_allclose(np.asarray(d.selected.delta),-np.linalg.solve(t.hessian,t.gradient),atol=1e-12)
    assert np.isclose(d.selected.response,t.center+.5*t.gradient@d.selected.delta)
    assert np.isclose(d.selected.col_local,d.selected.col+d.selected.delta[0])
    assert np.isclose(d.selected.row_local,d.selected.row+d.selected.delta[1])
    assert np.isclose(d.selected.layer_local,d.selected.layer+d.selected.delta[2])
    assert abs(d.selected.response)>=.008
    edge=next(p for p in d.filtered if p.status=="edge")
    h=sift.calculate_spatial_hessian(d.pyramid[edge.octave].dogs,edge.layer,edge.row,edge.col)
    tr=float(np.trace(h));det=float(np.linalg.det(h));ratio=tr*tr/det if det>1e-15 else float('inf')
    assert not (det>1e-15 and ratio<12.1)
    low=next(p for p in d.filtered_b if p.status=="low contrast")
    assert abs(low.response)<.008
    info=d.orientation
    raw_hist=np.zeros(36)
    for row,col,mag,angle,spatial,weight in info.contributions:
        assert np.isclose(weight,mag*spatial)
        u=angle/10;lo=int(np.floor(u))%36;f=u-np.floor(u)
        raw_hist[lo]+=weight*(1-f);raw_hist[(lo+1)%36]+=weight*f
    np.testing.assert_allclose(raw_hist,info.raw_histogram,atol=1e-12)
    smooth=(np.roll(raw_hist,2)+4*np.roll(raw_hist,1)+6*raw_hist+
            4*np.roll(raw_hist,-1)+np.roll(raw_hist,-2))/16
    np.testing.assert_allclose(smooth,info.histogram,atol=1e-12)
    peak=float(np.max(smooth))
    peak_indices=[i for i,value in enumerate(smooth) if peak>0 and
                  value>=.8*peak and value>smooth[(i-1)%36] and
                  value>=smooth[(i+1)%36]]
    peak_indices.sort(key=lambda i:(-smooth[i],i))
    assert tuple(10.0*i for i in peak_indices)==info.peaks
    assert d.oriented.orientation in info.peaks
    locations={(p.octave,round(p.row_global,5),round(p.col_global,5))
               for p in d.described}
    assert len(d.described)>len(locations)
    assert len(locations)<=sum(p.status=="kept" for p in d.filtered)
    desc=d.descriptor
    independently=np.zeros((4,4,8))
    for row,col,u,v,angle,weight in desc.samples:
        b=angle/45
        for rr in (int(np.floor(v)),int(np.floor(v))+1):
            for cc in (int(np.floor(u)),int(np.floor(u))+1):
                for bb in (int(np.floor(b)),int(np.floor(b))+1):
                    if 0<=rr<4 and 0<=cc<4:
                        independently[rr,cc,bb%8]+=(weight*(1-abs(v-rr))*
                                                        (1-abs(u-cc))*(1-abs(b-bb)))
    np.testing.assert_allclose(independently,desc.cells,atol=1e-12)
    np.testing.assert_allclose(desc.raw,independently.reshape(128),atol=1e-12)
    for r in range(4):
        for c in range(4):
            for b in range(8): assert sift.descriptor_index(r,c,b)==(r*4+c)*8+b
    norm=np.linalg.norm(desc.raw)
    np.testing.assert_allclose(desc.first_l2,desc.raw/norm)
    np.testing.assert_allclose(desc.clipped,np.minimum(desc.first_l2,.2))
    np.testing.assert_allclose(desc.final,desc.clipped/np.linalg.norm(desc.clipped))
    m=d.accepted;q=d.described[m.query].descriptor
    distances=np.array([np.sqrt(np.sum((q-p.descriptor)**2)) for p in d.described_b])
    order=np.argsort(distances,kind="stable")
    assert (m.nearest,m.second)==tuple(map(int,order[:2]))
    assert np.isclose(m.d1,distances[m.nearest]) and np.isclose(m.d2,distances[m.second])
    assert np.isclose(m.ratio,m.d1/m.d2) and m.ratio<.8
    if d.rejected is not None: assert d.rejected.ratio>=.8
    source=d.described[m.query];target=d.described_b[m.nearest]
    expected=d.affine@np.array([source.col_global,source.row_global,1.])
    assert np.isclose(d.geometry_error,np.hypot(expected[0]-target.col_global,
                                               expected[1]-target.row_global))
    false_matches=[(sift.evaluate_geometric_correspondence(d.described[z.query],
                    d.described_b[z.nearest],d.affine)[0],z)
                   for z in sift.match_descriptors(d.described,d.described_b)
                   if z.accepted]
    assert any(error>4 and z.ratio<.8 for error,z in false_matches)
    assert 0<=source.row_global<129 and 0<=source.col_global<129
    assert 0<=target.row_global<129 and 0<=target.col_global<129
    print("NUMERIC OK",len(d.raw),Counter(p.status for p in d.filtered),
          len(d.described),len(d.described_b),"match error",d.geometry_error)


def preflight():
    from manim import tempconfig
    from sift_deep_dive import SIFTDeepDive
    with tempconfig({"skip_animations":True,"save_last_frame":True,"quality":"low_quality"}):
        scene=SIFTDeepDive()
        scene.render()
        assert not any(m.updaters for root in scene.mobjects for m in root.get_family())
    print("LAYOUT OK: in-frame finite geometry, no leaked updaters; "
          f"timeline={scene.renderer.time:.2f}s")
    for (name,start),next_start in zip(scene.storyboard,
            [point[1] for point in scene.storyboard[1:]]+[scene.renderer.time]):
        print(f"STORYBOARD {start:6.1f}s +{next_start-start:5.1f}s {name}")


def video(path: Path, contact=False):
    import av
    from PIL import Image,ImageDraw
    assert path.is_file() and path.stat().st_size>10000,path
    frames=[];count=0;dark=0;longest=0
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        fps=float(stream.average_rate); dims=(stream.width,stream.height)
        stride=max(1,round(fps*22))
        for frame in container.decode(video=0):
            count+=1
            if count==1 or count%stride==0:
                frames.append((count/fps,frame.to_image().copy()))
            gray=cv2.cvtColor(frame.to_ndarray(format="rgb24"),cv2.COLOR_RGB2GRAY)
            dark=dark+1 if float(np.std(gray))<1.5 else 0
            longest=max(longest,dark)
    duration=count/fps
    assert dims==((854,480) if path==REVIEW else (1920,1080))
    assert abs(fps-(15 if path==REVIEW else 30))<.01
    assert 420<=duration<=570,(duration,path)
    assert longest/fps<2
    if contact:
        cols=3;rows=(len(frames)+cols-1)//cols
        sheet=Image.new("RGB",(cols*427,rows*270),"#0B1020")
        draw=ImageDraw.Draw(sheet)
        for i,(seconds,img) in enumerate(frames):
            x=i%cols*427;y=i//cols*270
            sheet.paste(img.resize((427,240)),(x,y))
            draw.text((x+8,y+243),f"{seconds:.1f}s",fill="white")
        output=HERE/"media/sift_deep_dive_contact.png"
        sheet.save(output)
        print("contact",output)
    print(f"VIDEO OK {path}: {duration:.2f}s, {fps:g}fps, {count} decoded frames, {dims}")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--video",choices=("review","final"))
    parser.add_argument("--contact-sheet",action="store_true")
    args=parser.parse_args()
    validate()
    if args.preflight: preflight()
    if args.video: video(REVIEW if args.video=="review" else FINAL,args.contact_sheet)
