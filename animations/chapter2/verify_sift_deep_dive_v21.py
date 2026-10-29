"""Independent checks, preflight and full video decode for SIFT Deep Dive V2.1."""

from __future__ import annotations

import argparse
from math import atan2, floor, hypot
from pathlib import Path

import av
import cv2
import numpy as np

from sift_deep_dive_data import build_data, gaussian_pixel_example
from sift_deep_dive_v21 import (descriptor_sample_factors, descriptor_terms,
                               orientation_vote, trilinear_display)
from verify_sift_deep_dive import validate as validate_v2
from utils import scale_space, sift

HERE=Path(__file__).resolve().parent
REVIEW=HERE/'media/videos/sift_deep_dive_v21/480p15/SIFTDeepDive_V21_review_polished.mp4'
FINAL=HERE/'media/videos/sift_deep_dive_v21/1080p30/SIFTDeepDive_V21_final_polished.mp4'


def validate():
    validate_v2()
    d=build_data(); kp=d.candidate
    kernel=scale_space.gaussian_kernel(1.6)
    patch, actual_kernel, total=gaussian_pixel_example(d,50,35)
    assert kernel.shape==patch.shape==(15,15)
    np.testing.assert_allclose(kernel,actual_kernel,atol=0)
    assert np.isclose(kernel.sum(),1.0,atol=1e-12)
    assert np.isclose(np.sum(patch*kernel),total,atol=1e-12)
    assert np.isclose(total,d.pyramid[0].gaussians[0][50,35],atol=1e-12)
    dogs=d.pyramid[kp.octave].dogs
    values=np.array([dogs[j][kp.row,kp.col] for j in range(5)])
    assert (kp.row,kp.col)==(33,103) and kp.layer==1
    assert values[1]<values[0] and values[1]<values[2]
    assert np.isclose(values[1],kp.dog_value)
    for j in range(5):
        assert np.isclose(values[j],dogs[j][33,103])
    events=scale_space.inspect_neighbors(dogs,kp.layer,kp.row,kp.col)
    assert len(events)==26
    assert sum(e.layer==kp.layer for e in events)==8
    assert sum(e.layer==kp.layer-1 for e in events)==9
    assert sum(e.layer==kp.layer+1 for e in events)==9
    assert all(kp.dog_value<e.value for e in events)
    assert d.selected.col==102 and d.selected.col_local>102
    selected=d.selected; info=d.orientation
    image,j=sift.gaussian_level(selected,d.pyramid[selected.octave])
    gx,gy=sift.gradient_field(image)
    raw=np.zeros(36)
    for sample in info.contributions:
        row,col,mag,angle,spatial,weight=sample
        ix=(image[row,col+1]-image[row,col-1])/2
        iy=(image[row+1,col]-image[row-1,col])/2
        assert np.isclose(ix,gx[row,col]) and np.isclose(iy,gy[row,col])
        assert np.isclose(mag,hypot(ix,iy))
        assert np.isclose(angle,np.degrees(atan2(iy,ix))%360)
        assert np.isclose(weight,mag*spatial)
        b0,b1,v0,v1=orientation_vote(sample)
        assert b0==int(floor(angle/10))%36 and b1==(b0+1)%36
        assert np.isclose(v0+v1,weight)
        raw[b0]+=v0;raw[b1]+=v1
        # Image y-down maps to screen y-up with one sign reversal.
        assert np.isclose(np.arctan2(-iy,ix),
                          np.arctan2(-np.sin(np.radians(angle)),
                                     np.cos(np.radians(angle))))
    np.testing.assert_allclose(raw,info.raw_histogram,atol=1e-12)
    smooth=(np.roll(raw,2)+4*np.roll(raw,1)+6*raw+
            4*np.roll(raw,-1)+np.roll(raw,-2))/16
    np.testing.assert_allclose(smooth,info.histogram,atol=1e-12)
    peak=int(np.argmax(smooth)); assert peak*10==info.peaks[0]
    x0,width=.05,6.65
    assert np.isclose(x0+(peak+.5)*width/36,
                      x0+(info.peaks[0]/10+.5)*width/36)
    desc=d.descriptor
    rebuilt=np.zeros((4,4,8))
    for sample in desc.samples:
        for rr,cc,bb,wy,wx,wb,part in descriptor_terms(sample):
            assert np.isclose(part,sample[5]*wy*wx*wb)
            rebuilt[rr,cc,bb]+=part
    np.testing.assert_allclose(rebuilt,desc.cells,atol=1e-12)
    np.testing.assert_allclose(rebuilt.reshape(128),desc.raw,atol=1e-12)
    index=int(np.argmax(desc.raw))
    r,c,b=np.unravel_index(index,(4,4,8))
    assert sift.descriptor_index(int(r),int(c),int(b))==index
    pieces=[term[-1] for sample in desc.samples for term in descriptor_terms(sample)
            if term[:3]==(r,c,b)]
    assert len(pieces)>4 and np.isclose(sum(pieces),desc.raw[index])
    selected_sample=sorted(
        ((term[-1],sample) for sample in desc.samples
         for term in descriptor_terms(sample) if term[:3]==(r,c,b)),
        key=lambda pair:-pair[0])[0][1]
    magnitude,gaussian,weighted=descriptor_sample_factors(
        selected_sample,d.oriented,d.pyramid[d.oriented.octave])
    row,col=selected_sample[:2]
    image,_=sift.gaussian_level(d.oriented,d.pyramid[d.oriented.octave])
    gx,gy=sift.gradient_field(image)
    assert np.isclose(magnitude,np.hypot(gx[row,col],gy[row,col]))
    width=3*d.oriented.sigma_local
    dx,dy=col-d.oriented.col_local,row-d.oriented.row_local
    assert np.isclose(gaussian,np.exp(-(dx*dx+dy*dy)/(2*(2*width)**2)))
    assert np.isclose(weighted,magnitude*gaussian)
    assert abs(float(f'{magnitude:.9f}')*float(f'{gaussian:.9f}')-
               float(f'{weighted:.9f}'))<1.5e-9
    terms=descriptor_terms(selected_sample)
    assert len(terms)==8
    assert terms[0][:3]==(1,1,0) and terms[4][:3]==(2,1,0)
    display_sum=0.0
    for rr,cc,bb,wy,wx,wtheta,part in terms:
        term=(rr,cc,bb,wy,wx,wtheta,part)
        factors,result=trilinear_display(selected_sample,term)
        shown=np.array([float(factor) for factor in factors])
        assert np.all(shown>0) and float(result)>0
        np.testing.assert_allclose(shown,[weighted,wx,wy,wtheta],atol=5.1e-10)
        assert abs(np.prod(shown)-float(result))<1.5e-9
        assert np.isclose(part,weighted*wx*wy*wtheta,atol=1e-12)
        assert sift.descriptor_index(rr,cc,bb)==(rr*4+cc)*8+bb
        display_sum+=part
    assert np.isclose(display_sum,weighted,atol=1e-12)
    np.testing.assert_allclose(desc.cells.reshape(128),desc.raw,atol=0)
    np.testing.assert_allclose(desc.first_l2,desc.raw/np.linalg.norm(desc.raw))
    np.testing.assert_allclose(desc.clipped,np.minimum(desc.first_l2,.2))
    np.testing.assert_allclose(desc.final,desc.clipped/np.linalg.norm(desc.clipped))
    m=d.accepted; q=d.described[m.query].descriptor
    distance=np.linalg.norm(np.stack([p.descriptor for p in d.described_b])-q,axis=1)
    order=np.argsort(distance,kind='stable')
    assert (m.nearest,m.second)==tuple(map(int,order[:2]))
    assert np.isclose(distance[m.nearest],m.d1)
    assert np.isclose(distance[m.second],m.d2)
    assert np.isclose(m.ratio,m.d1/m.d2) and m.ratio<.8
    print('V21 NUMERIC OK: Gaussian 15x15/225, DoG, 26 neighbors, '
          'gradient/votes, 90-degree peak, 8 displayed trilinear products, '
          '128D, normalization, matching')


def preflight():
    from manim import tempconfig
    from sift_deep_dive_v21 import SIFTDeepDiveV21
    with tempconfig({'skip_animations':True,'save_last_frame':True,
                     'quality':'low_quality','progress_bar':'none'}):
        scene=SIFTDeepDiveV21();scene.render()
    assert not any(m.updaters for root in scene.mobjects for m in root.get_family())
    assert 600<=scene.renderer.time<=780,scene.renderer.time
    print(f'V21 PREFLIGHT OK: {scene.renderer.time:.2f}s, in frame, no updaters')
    for (name,start),end in zip(scene.storyboard,
                                [v[1] for v in scene.storyboard[1:]]+
                                [scene.renderer.time]):
        print(f'STORYBOARD {start:6.1f}s +{end-start:5.1f}s {name}')


def video(path, contact=False):
    from PIL import Image,ImageDraw
    assert path.is_file() and path.stat().st_size>10000
    samples=[]; count=0;blank=0;longest=0
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        fps=float(stream.average_rate)
        size=(stream.width,stream.height)
        stride=round(fps*24)
        for frame in container.decode(video=0):
            count+=1
            if count==1 or count%stride==0:
                samples.append((count/fps,frame.to_image().copy()))
            gray=cv2.cvtColor(frame.to_ndarray(format='rgb24'),cv2.COLOR_RGB2GRAY)
            blank=blank+1 if gray.std()<1.5 else 0
            longest=max(longest,blank)
    duration=count/fps
    assert size==((854,480) if path==REVIEW else (1920,1080)),size
    assert abs(fps-(15 if path==REVIEW else 30))<.01
    assert 600<=duration<=780,duration
    assert longest/fps<2
    if contact:
        cols=3;rows=(len(samples)+cols-1)//cols
        sheet=Image.new('RGB',(cols*427,rows*270),'#0B1020')
        draw=ImageDraw.Draw(sheet)
        for k,(second,frame) in enumerate(samples):
            x=k%cols*427;y=k//cols*270
            sheet.paste(frame.resize((427,240)),(x,y))
            draw.text((x+8,y+243),f'{second:.1f}s',fill='white')
        out=HERE/'media/sift_deep_dive_v21_polished_contact.png'
        sheet.save(out)
        print('CONTACT',out)
    print(f'VIDEO OK {path}: {duration:.2f}s, {fps:g} fps, '
          f'{count} decoded frames, {size}')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--video',choices=('review','final'))
    parser.add_argument('--contact-sheet',action='store_true')
    args=parser.parse_args()
    validate()
    if args.preflight: preflight()
    if args.video: video(REVIEW if args.video=='review' else FINAL,args.contact_sheet)
