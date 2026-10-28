"""Independent Harris arithmetic, layout preflight, and complete MP4 decode."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from chapter2_common import HERE
from harris_corner import HarrisCornerDetection, build_harris_fixture
from utils import harris

REVIEW = HERE / "media/videos/harris_corner/480p15/HarrisCornerDetection_review_v6.mp4"
FINAL = HERE / "media/videos/harris_corner/1080p60/HarrisCornerDetection_final_v4.mp4"


def validate():
    data=build_harris_fixture()
    flat,edge,corner=data.images
    assert np.all(flat==.35)
    assert np.array_equal(edge[:,20],np.r_[np.zeros(20),np.ones(21)])
    assert corner[19,19]==0 and corner[20,20]==1
    before,after=harris.shifted_patches(corner,20,20,9,1,1)
    assert before.shape==after.shape==(9,9)
    assert np.sum((after-before)**2)==data.energy[3,3]==11
    for v in range(-2,3):
        for u in range(-2,3):
            p=corner[20-4:20+5,20-4:20+5]
            q=corner[20+v-4:20+v+5,20+u-4:20+u+5]
            np.testing.assert_allclose(np.sum((q-p)**2),data.energy[v+2,u+2])
    # Interior normalized Sobel correlation, derived directly from kernel samples.
    p=corner[19:22,19:22]
    ix=float(np.sum(p*harris.KX)); iy=float(np.sum(p*harris.KY))
    assert np.isclose(ix,data.pipelines[2].ix[20,20])
    assert np.isclose(iy,data.pipelines[2].iy[20,20])
    kernel=harris.integration_kernel(5,1.2)
    assert np.isclose(kernel.sum(),1)
    for pipe in data.pipelines:
        ix,iy=pipe.ix,pipe.iy
        products=(ix*ix,ix*iy,iy*iy)
        vals=[float(np.sum(kernel*product[18:23,18:23])) for product in products]
        matrix=np.array([[vals[0],vals[1]],[vals[1],vals[2]]])
        np.testing.assert_allclose(matrix,harris.tensor_at(pipe.maps,20,20),atol=1e-14)
        assert np.allclose(matrix,matrix.T)
        eig=np.linalg.eigvalsh(matrix)[::-1]
        np.testing.assert_allclose(eig,[pipe.large[20,20],pipe.small[20,20]],atol=1e-14)
        expected=np.linalg.det(matrix)-.05*np.trace(matrix)**2
        np.testing.assert_allclose(expected,pipe.harris[20,20],atol=1e-14)
        np.testing.assert_allclose(eig[-1],pipe.shi_tomasi[20,20],atol=1e-14)
    assert data.pipelines[0].large[20,20]<.005
    assert data.pipelines[1].large[20,20]>.005 and data.pipelines[1].small[20,20]<.005
    assert data.pipelines[2].small[20,20]>.005
    response=data.pipelines[2].harris
    assert np.isclose(data.threshold,response.max()*.08)
    independent=np.zeros_like(data.keep_mask)
    for r in range(2,39):
        for c in range(2,39):
            v=response[r,c]
            patch=response[r-2:r+3,c-2:c+3]
            if v>data.threshold and v==patch.max():
                equal=np.argwhere(patch==v)[0]
                independent[r,c]=tuple(equal)==(2,2)
    np.testing.assert_array_equal(independent,data.keep_mask)
    for rc,expected in ((data.keep_example,True),(data.reject_example,False)):
        assert harris.nms_decision(response,data.threshold,5,*rc)[0] is expected
    assert np.argwhere(independent).tolist()==[[20,20]]
    print("Harris numeric OK: shift energy, Sobel /8, tensor, eigenvalues, R, threshold, 5x5 tie-breaking, final corner")


def preflight():
    from manim import tempconfig
    with tempconfig({"skip_animations":True,"save_last_frame":True,"quality":"low_quality"}):
        scene=HarrisCornerDetection()
        scene.render()
        assert not any(m.updaters for root in scene.mobjects for m in root.get_family())
    print("Harris layout preflight OK: 16:9, finite in-frame objects, no updaters")


def video(path, contact=False):
    import av
    import PIL.Image
    from PIL import ImageDraw
    assert path.is_file() and path.stat().st_size>10000, path
    selected=[]; count=0; dark_run=0; longest_dark=0
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        fps=float(stream.average_rate)
        dims=(stream.width,stream.height)
        for frame in container.decode(video=0):
            count+=1
            if count%max(1,round(fps*5))==1:
                selected.append((count/fps,frame.to_image().copy()))
            gray=cv2.cvtColor(frame.to_ndarray(format="rgb24"),cv2.COLOR_RGB2GRAY)
            dark_run=dark_run+1 if float(np.std(gray))<1.5 else 0
            longest_dark=max(longest_dark,dark_run)
    duration=count/fps
    assert dims==(854,480) if path==REVIEW else dims==(1920,1080)
    assert abs(fps-(15 if path==REVIEW else 60))<.01
    assert 48<=duration<=68,(duration,path)
    assert longest_dark/fps<2,(longest_dark,fps)
    if contact:
        tiles=[img.resize((427,240)) for _,img in selected]
        cols=3; rows=(len(tiles)+cols-1)//cols
        sheet=PIL.Image.new("RGB",(cols*427,rows*270),"#0B1020")
        draw=ImageDraw.Draw(sheet)
        for i,(time,img) in enumerate(selected):
            x=(i%cols)*427;y=(i//cols)*270
            sheet.paste(tiles[i],(x,y))
            draw.text((x+8,y+243),f"{time:.1f}s",fill="white")
        out=HERE/"media/harris_review_contact.png"
        sheet.save(out)
        print("contact",out)
    print(f"Harris video OK: {path}, {duration:.2f}s, {fps:g} fps, {count} decoded frames, {dims}, {path.stat().st_size} bytes")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--video",choices=("review","final"))
    parser.add_argument("--contact-sheet",action="store_true")
    args=parser.parse_args()
    validate()
    if args.preflight: preflight()
    if args.video: video(REVIEW if args.video=="review" else FINAL,args.contact_sheet)
